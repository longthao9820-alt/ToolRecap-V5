"""Single-click entrypoint for ToolRecap V5 Desktop Application."""

import os
import sys


def setup_frozen_tcl() -> None:
    """Ensure Tcl/Tk data directories are found when running frozen."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return

    base_mei = getattr(sys, "_MEIPASS", None)
    base_dirs: list[str] = []
    if base_mei:
        base_dirs.extend([base_mei, os.path.join(base_mei, "_internal")])
    exe_dir = os.path.dirname(sys.executable)
    base_dirs.extend([exe_dir, os.path.join(exe_dir, "_internal")])

    for b in base_dirs:
        if "TCL_LIBRARY" not in os.environ:
            for candidate in [
                os.path.join(b, "_tcl_data"),
                os.path.join(b, "tcl", "tcl8.6"),
                os.path.join(b, "tcl8.6"),
                os.path.join(b, "tcl"),
            ]:
                if os.path.isdir(candidate):
                    os.environ["TCL_LIBRARY"] = candidate.replace("\\", "/")
                    break

        if "TK_LIBRARY" not in os.environ:
            for candidate in [
                os.path.join(b, "_tk_data"),
                os.path.join(b, "tcl", "tk8.6"),
                os.path.join(b, "tk8.6"),
                os.path.join(b, "tk"),
            ]:
                if os.path.isdir(candidate):
                    os.environ["TK_LIBRARY"] = candidate.replace("\\", "/")
                    break


setup_frozen_tcl()


def main() -> None:
    if "--update-handshake" in sys.argv:
        print("TOOLRECAP_V5_HANDSHAKE_OK")
        # Retain the old marker so an already-installed V4 updater can verify
        # the first V5 executable during an in-place upgrade.
        print("TOOLRECAP_V4_HANDSHAKE_OK")
        raise SystemExit(0)

    if "--update-helper" in sys.argv:
        from toolrecap_v4.updater.helper import main as run_update_helper

        run_update_helper()
        return

    if "--selfcheck" in sys.argv:
        from toolrecap_v4.selfcheck import run_selfcheck

        sys.exit(run_selfcheck())

    from toolrecap_v4.ui.main_window import run_app

    run_app()


if __name__ == "__main__":
    main()
