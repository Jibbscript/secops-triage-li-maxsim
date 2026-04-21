from __future__ import annotations

import pytest

from alert_triage.triage import RetryPolicy
from alert_triage.triage.retry import retry_call


def test_retry_policy_default_is_one_shot() -> None:
    attempts = {"count": 0}

    def operation() -> str:
        attempts["count"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        retry_call(
            operation,
            policy=RetryPolicy(),
            is_retryable=lambda exc: True,
        )

    assert attempts["count"] == 1


def test_retry_call_applies_deterministic_backoff_schedule() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    result = retry_call(
        operation,
        policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0.25,
            max_backoff_seconds=1.0,
            backoff_multiplier=2.0,
        ),
        is_retryable=lambda exc: True,
        sleep_fn=sleeps.append,
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeps == [0.25, 0.5]


def test_retry_call_stops_on_non_retryable_failure() -> None:
    attempts = {"count": 0}

    def operation() -> str:
        attempts["count"] += 1
        raise RuntimeError("not retryable")

    with pytest.raises(RuntimeError, match="not retryable"):
        retry_call(
            operation,
            policy=RetryPolicy(max_attempts=4, initial_backoff_seconds=0.1),
            is_retryable=lambda exc: False,
            sleep_fn=lambda _: None,
        )

    assert attempts["count"] == 1
