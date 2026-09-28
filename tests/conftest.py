"""Pytest test configuration and fixtures."""

import gc
import os
from pathlib import Path
import tempfile
import pytest

# Ensure clean temp root on Windows to prevent dead symlink PermissionError in sessionfinish
_kilo_tmp = Path(tempfile.gettempdir()) / "kilo" / "pytest"
_kilo_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_kilo_tmp))


@pytest.fixture(autouse=True)
def _cleanup_tk_cycles():
    """Ensure cyclic garbage is collected synchronously on main thread after each test."""
    yield
    gc.collect()
