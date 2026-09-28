"""Tests for cancellation primitives."""

import pytest

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError


def test_token_initial_state() -> None:
    """Verify initial token state is not cancelled."""
    token = CancellationToken()
    assert not token.is_cancelled
    # Should not raise
    token.check_cancelled()


def test_token_cancel() -> None:
    """Verify cancellation sets is_cancelled and check_cancelled raises CancelledError."""
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled

    with pytest.raises(CancelledError, match="Operation was cancelled"):
        token.check_cancelled()


def test_cancellation_callbacks_executed() -> None:
    """Verify registered callbacks are executed on cancel()."""
    token = CancellationToken()
    called = []

    token.register_callback(lambda: called.append("cb1"))
    token.register_callback(lambda: called.append("cb2"))

    assert len(called) == 0
    token.cancel()
    assert called == ["cb1", "cb2"]


def test_callback_executed_immediately_if_already_cancelled() -> None:
    """Verify callback is executed immediately if registered after cancellation."""
    token = CancellationToken()
    token.cancel()

    called = []
    token.register_callback(lambda: called.append("immediate"))
    assert called == ["immediate"]


def test_callback_exception_does_not_prevent_other_callbacks() -> None:
    """Verify an exception in one callback does not prevent others from running."""
    token = CancellationToken()
    called = []

    def faulty_callback():
        raise RuntimeError("callback failure")

    token.register_callback(faulty_callback)
    token.register_callback(lambda: called.append("recovered"))

    token.cancel()
    assert token.is_cancelled
    assert called == ["recovered"]
