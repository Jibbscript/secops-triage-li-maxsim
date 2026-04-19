from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .base import BINARY_BITS, BinaryCalibratorModel


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
        rotation = BinaryCalibratorIO._normalize_rotation(data["rotation"])
        calibrator = BinaryCalibratorModel(
            thresholds=data["thresholds"],
            rotation=rotation,
        )
        stored_encoder_id = str(data["encoder_id"][0])
        if expected_encoder_id is not None:
            if stored_encoder_id != expected_encoder_id:
                raise ValueError(
                    f"encoder_id mismatch: stored={stored_encoder_id!r} expected={expected_encoder_id!r}"
                )
        stored_checksum = str(data["checksum"][0])
        expected_checksum = BinaryCalibratorIO.checksum(calibrator, stored_encoder_id)
        if stored_checksum != expected_checksum:
            raise ValueError(
                "calibrator checksum mismatch: "
                f"stored={stored_checksum!r} expected={expected_checksum!r}"
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

    @staticmethod
    def _normalize_rotation(rotation: np.ndarray) -> np.ndarray | None:
        if rotation.shape == () and rotation.dtype == object and rotation.item() is None:
            return None
        if rotation.size == 0:
            return None
        rotation = np.asarray(rotation, dtype=np.float32)
        if rotation.shape != (BINARY_BITS, BINARY_BITS):
            raise ValueError(
                "rotation matrix shape mismatch: "
                f"expected {(BINARY_BITS, BINARY_BITS)} got {rotation.shape}"
            )
        return rotation
