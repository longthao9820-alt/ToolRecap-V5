"""Tests for Windows notification service and honest API vs visual receipt tracking."""

import sys
from unittest.mock import MagicMock, patch
import pytest

from toolrecap_v4.ui.notifications import NotificationResult, WindowsNotificationService


def test_notification_result_distinguishes_api_and_visual_receipt():
    """Verify NotificationResult cleanly distinguishes API dispatch from visual receipt."""
    res = NotificationResult(api_success=True, visual_receipt=None, sound_played=True, taskbar_flashed=True)
    assert res.api_success is True
    # Visual receipt is None (unconfirmed fire-and-forget), NOT assumed True
    assert res.visual_receipt is None
    assert res.sound_played is True
    assert res.taskbar_flashed is True
    assert res.error is None


def test_notification_service_terminal_completed():
    """Verify notify_project_completed constructs accurate terminal notification."""
    service = WindowsNotificationService()

    with patch.object(service, "send_toast") as mock_send:
        mock_send.return_value = NotificationResult(api_success=True, visual_receipt=None, sound_played=True)

        res = service.notify_project_completed(
            project_name="Yellowstone_S01",
            outputs_count=3,
            sound=True,
        )

        assert res.api_success is True
        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        assert "Hoàn tất dự án" in kwargs.get("title", "")
        assert "Yellowstone_S01" in kwargs.get("body", "")
        assert "3 video" in kwargs.get("body", "")
        assert kwargs.get("sound") is True


def test_notification_service_terminal_failed():
    """Verify notify_project_failed constructs accurate terminal error notification."""
    service = WindowsNotificationService()

    with patch.object(service, "send_toast") as mock_send:
        mock_send.return_value = NotificationResult(api_success=True, visual_receipt=None, sound_played=True)

        res = service.notify_project_failed(
            project_name="Yellowstone_S01",
            error_msg="VoiceStudio connection refused",
            sound=True,
        )

        assert res.api_success is True
        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        assert "Cần kiểm tra" in kwargs.get("title", "")
        assert "VoiceStudio connection refused" in kwargs.get("body", "")


def test_notification_service_update_check():
    """Verify notify_update_check explicitly reports not configured."""
    service = WindowsNotificationService()

    with patch.object(service, "send_toast") as mock_send:
        mock_send.return_value = NotificationResult(api_success=True, visual_receipt=None)

        res = service.notify_update_check()
        assert res.api_success is True
        args, kwargs = mock_send.call_args
        assert "Chưa cấu hình máy chủ cập nhật" in kwargs.get("body", "")


def test_send_toast_with_callbacks_records_visual_receipt():
    """Verify that when toast activation/dismissal callbacks fire, visual_receipt becomes True."""
    service = WindowsNotificationService()
    if service._toaster is None:
        pytest.skip("windows-toasts not available on this platform")

    toast_instances = []

    def mock_show_toast(toast):
        toast_instances.append(toast)
        # Simulate OS calling on_activated
        if toast.on_activated:
            toast.on_activated(None)

    with patch.object(service._toaster, "show_toast", side_effect=mock_show_toast):
        user_called = False

        def on_user_click():
            nonlocal user_called
            user_called = True

        res = service.send_toast(
            title="Test Title",
            body="Test Body",
            on_activated=on_user_click,
        )

        assert res.api_success is True
        assert res.visual_receipt is True
        assert user_called is True
