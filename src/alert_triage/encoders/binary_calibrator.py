from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from alert_triage.encoders.base import BinaryCalibratorModel


class BinaryCalibratorIO:
    """Persistence helpers for BinaryCalibratorModel sidecar artifacts."""

    @staticmethod
    def save(path: str | Path, calibrator: BinaryCalibratorModel, *, encoder_id: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thresholds": calibrator.thresholds,
            "rotation": calibrator.rotation,
            "encoder_id": np.array([encoder_id]),
            "checksum": np.array(
                [BinaryCalibratorIO.checksum(calibrator, encoder_id)], dtype=object
            ),
        }
        np.savez_compressed(out, **payload)

    @staticmethod
    def load(path: str | Path, *, expected_encoder_id: str | None = None) -> BinaryCalibratorModel:
        data = np.load(Path(path), allow_pickle=True)
        calibrator = BinaryCalibratorModel(
            thresholds=data["thresholds"],
            rotation=data["rotation"] if "rotation" in data and data["rotation"].size else None,
        )
        if expected_encoder_id is not None:
            stored = str(data["encoder_id"][0])
            if stored != expected_encoder_id:
                raise ValueError(
                    f"encoder_id mismatch: stored={stored!r} expected={expected_encoder_id!r}"
                )
        return calibrator

    @staticmethod
    def checksum(calibrator: BinaryCalibratorModel, encoder_id: str) -> str:
        h = hashlib.sha256()
        h.update(encoder_id.encode("utf-8"))
        h.update(np.asarray(calibrator.thresholds, dtype=np.float32).tobytes())
        if calibrator.rotation is not None:
            h.update(np.asarray(calibrator.rotation, dtype=np.float32).tobytes())
        return h.hexdigest()
