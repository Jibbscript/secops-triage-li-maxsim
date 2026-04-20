from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from alert_triage.encoders.base import EncodedTokens


@dataclass
class InMemoryTokenVectorStore:
    """Minimal checked-in store for the phase-1 fp16 baseline slice."""

    _fp16: dict[str, np.ndarray] = field(default_factory=dict)
    _bin: dict[str, np.ndarray] = field(default_factory=dict)

    def upsert(self, alert_ids: Sequence[str], docs: Sequence[EncodedTokens]) -> None:
        if len(alert_ids) != len(docs):
            raise ValueError("alert_ids and docs must have matching lengths")

        for alert_id, doc in zip(alert_ids, docs, strict=True):
            if doc.fp16 is not None:
                self._fp16[alert_id] = np.asarray(doc.fp16, dtype=np.float32)
            if doc.binary is not None:
                self._bin[alert_id] = np.asarray(doc.binary, dtype=np.uint8)

    def fetch_fp16(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        return {alert_id: self._fp16[alert_id] for alert_id in ids if alert_id in self._fp16}

    def fetch_bin(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        return {alert_id: self._bin[alert_id] for alert_id in ids if alert_id in self._bin}

    def iter_fp16(self) -> list[tuple[str, np.ndarray]]:
        return [(alert_id, self._fp16[alert_id]) for alert_id in sorted(self._fp16)]

    def size(self) -> int:
        return len(self._fp16)
