"""Windows notification service for ToolRecap V4.

Invariants:
- Real Windows toast notifications via windows-toasts (WinRT).
- Sound (winsound) and taskbar flash (FlashWindowEx) flags.
- Sent only on terminal project events (completion, fatal failure, update check).
- Zero spam: no notifications for clips, segments, or transient retries.
- Zero worker dialogs: worker thread never opens UI dialogs.
- Distinguishes API call success from visual receipt.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Windows FlashWindowEx constants
FLASHW_STOP = 0
FLASHW_CAPTION = 1
FLASHW_TRAY = 2
FLASHW_ALL = 3
FLASHW_TIMER = 4
FLASHW_TIMERNOFG = 12


class FLASHWINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("hwnd", wintypes.HWND),
        ("dwFlags", wintypes.DWORD),
        ("uCount", wintypes.UINT),
        ("dwTimeout", wintypes.DWORD),
    ]


@dataclass
class NotificationResult:
    """Detailed result of an attempted notification.
    
    Distinguishes between API call success and visual receipt.
    """
    api_success: bool
    visual_receipt: Optional[bool] = None
    sound_played: bool = False
    taskbar_flashed: bool = False
    error: Optional[str] = None


class WindowsNotificationService:
    """Manages desktop toast notifications, audio alerts, and taskbar flashing on Windows."""

    def __init__(self, app_name: str = "ToolRecap V5") -> None:
        self.app_name = app_name
        self._toaster: Optional[Any] = None
        self._init_toaster()

    def _init_toaster(self) -> None:
        """Initialize the windows-toasts toaster if on Windows and library available."""
        if sys.platform != "win32":
            return
        try:
            from windows_toasts import WindowsToaster
            self._toaster = WindowsToaster(self.app_name)
        except Exception as e:
            logger.warning("Failed to initialize WindowsToaster: %s", e)
            self._toaster = None

    def play_sound(self) -> bool:
        """Play completion sound using Windows standard sound."""
        if sys.platform != "win32":
            return False
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
            return True
        except Exception as e:
            logger.warning("Failed to play sound: %s", e)
            return False

    def flash_taskbar(self, hwnd: Optional[int] = None) -> bool:
        """Flash the taskbar icon to alert the user when application is in background."""
        if sys.platform != "win32" or not hwnd:
            return False
        try:
            fwi = FLASHWINFO()
            fwi.cbSize = ctypes.sizeof(FLASHWINFO)
            fwi.hwnd = wintypes.HWND(hwnd)
            fwi.dwFlags = FLASHW_TRAY | FLASHW_TIMERNOFG
            fwi.uCount = 5
            fwi.dwTimeout = 0
            user32 = ctypes.windll.user32
            return bool(user32.FlashWindowEx(ctypes.byref(fwi)))
        except Exception as e:
            logger.warning("Failed to flash taskbar: %s", e)
            return False

    def send_toast(
        self,
        title: str,
        body: str,
        sound: bool = False,
        hwnd: Optional[int] = None,
        on_activated: Optional[Callable[[], None]] = None,
        on_dismissed: Optional[Callable[[], None]] = None,
    ) -> NotificationResult:
        """Send a Windows toast notification.
        
        Distinguishes API call success from visual receipt.
        """
        api_success = False
        visual_receipt: Optional[bool] = None
        sound_played = False
        taskbar_flashed = False
        err_msg: Optional[str] = None

        # Handle sound
        if sound:
            sound_played = self.play_sound()

        # Handle taskbar flash
        if hwnd:
            taskbar_flashed = self.flash_taskbar(hwnd)

        # Handle WinRT toast
        if self._toaster is not None:
            try:
                from windows_toasts import Toast

                toast = Toast()
                toast.text_fields = [title, body]

                if on_activated is not None:
                    def _wrap_activated(action_args: Any) -> None:
                        nonlocal visual_receipt
                        visual_receipt = True
                        try:
                            on_activated()
                        except Exception as e:
                            logger.error("Error in on_activated callback: %s", e)
                    toast.on_activated = _wrap_activated

                if on_dismissed is not None:
                    def _wrap_dismissed(reason: Any) -> None:
                        nonlocal visual_receipt
                        visual_receipt = True
                        try:
                            on_dismissed()
                        except Exception as e:
                            logger.error("Error in on_dismissed callback: %s", e)
                    toast.on_dismissed = _wrap_dismissed

                self._toaster.show_toast(toast)
                api_success = True
                # Visual receipt is unconfirmed until user actually sees/interacts with the toast
                if visual_receipt is None:
                    visual_receipt = None
            except Exception as e:
                err_msg = str(e)
                logger.warning("Toast notification via windows-toasts failed: %s", e)
                api_success = False
                visual_receipt = False
        else:
            err_msg = "windows-toasts not available or non-Windows system"
            api_success = False
            visual_receipt = False

        return NotificationResult(
            api_success=api_success,
            visual_receipt=visual_receipt,
            sound_played=sound_played,
            taskbar_flashed=taskbar_flashed,
            error=err_msg,
        )

    def notify_project_completed(
        self,
        project_name: str,
        outputs_count: int,
        sound: bool = True,
        hwnd: Optional[int] = None,
        on_open_output: Optional[Callable[[], None]] = None,
    ) -> NotificationResult:
        """Terminal notification: Project execution completed successfully."""
        title = "ToolRecap V5 — Hoàn tất dự án"
        body = f"Dự án '{project_name}' đã dựng thành công {outputs_count} video thành phẩm."
        return self.send_toast(
            title=title,
            body=body,
            sound=sound,
            hwnd=hwnd,
            on_activated=on_open_output,
        )

    def notify_project_failed(
        self,
        project_name: str,
        error_msg: str,
        sound: bool = True,
        hwnd: Optional[int] = None,
    ) -> NotificationResult:
        """Terminal notification: Project execution failed and requires user attention."""
        title = "ToolRecap V5 — Cần kiểm tra"
        body = f"Dự án '{project_name}' tạm dừng do lỗi: {error_msg[:120]}"
        return self.send_toast(
            title=title,
            body=body,
            sound=sound,
            hwnd=hwnd,
        )

    def notify_update_check(
        self,
        hwnd: Optional[int] = None,
    ) -> NotificationResult:
        """Terminal notification: Update check status."""
        title = "ToolRecap V5 — Cập nhật hệ thống"
        body = "Chưa cấu hình máy chủ cập nhật. Hệ thống đang hoạt động ở phiên bản hiện tại."
        return self.send_toast(
            title=title,
            body=body,
            sound=False,
            hwnd=hwnd,
        )
