from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    initial_backoff_seconds: float = 0.0
    max_backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("retry max_attempts must be >= 1")
        if self.initial_backoff_seconds < 0:
            raise ValueError("retry initial_backoff_seconds must be >= 0")
        if self.max_backoff_seconds < 0:
            raise ValueError("retry max_backoff_seconds must be >= 0")
        if self.backoff_multiplier < 1:
            raise ValueError("retry backoff_multiplier must be >= 1")

    def delay_before_attempt(self, attempt_number: int) -> float:
        if attempt_number <= 1:
            return 0.0
        delay = self.initial_backoff_seconds * (self.backoff_multiplier ** (attempt_number - 2))
        if self.max_backoff_seconds > 0:
            delay = min(delay, self.max_backoff_seconds)
        return delay


def retry_call(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
    on_retry: Callable[[Exception, int, float], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    attempt_number = 1
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempt_number >= policy.max_attempts or not is_retryable(exc):
                raise
            next_attempt = attempt_number + 1
            delay = policy.delay_before_attempt(next_attempt)
            if on_retry is not None:
                on_retry(exc, next_attempt, delay)
            if delay > 0:
                sleep_fn(delay)
            attempt_number = next_attempt
