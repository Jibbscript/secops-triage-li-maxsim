from __future__ import annotations

import numpy as np
import pytest

from alert_triage.encoders.base import BinaryCalibratorModel
from alert_triage.encoders.binary_calibrator import BinaryCalibratorIO


def test_fit_and_apply_emit_expected_shapes(rng: np.random.Generator) -> None:
    train = rng.normal(size=(32, 48)).astype(np.float32)
    tokens = rng.normal(size=(5, 48)).astype(np.float32)
    calibrator = BinaryCalibratorModel.fit(train)

    out = calibrator.apply(tokens)

    assert calibrator.thresholds.shape == (48,)
    assert calibrator.rotation is None
    assert out.shape == (5, 8)
    assert out.dtype == np.uint8


def test_fit_rejects_non_48_dim_inputs(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="expected 48-dim"):
        BinaryCalibratorModel.fit(rng.normal(size=(16, 49)).astype(np.float32))


def test_apply_rejects_non_48_dim_inputs(rng: np.random.Generator) -> None:
    calibrator = BinaryCalibratorModel.fit(rng.normal(size=(16, 48)).astype(np.float32))

    with pytest.raises(ValueError, match="expected 48-dim"):
        calibrator.apply(rng.normal(size=(4, 49)).astype(np.float32))


def test_save_load_roundtrip_preserves_none_rotation(tmp_path, rng: np.random.Generator) -> None:
    train = rng.normal(size=(32, 48)).astype(np.float32)
    calibrator = BinaryCalibratorModel.fit(train)
    path = tmp_path / "calibrator.npz"

    BinaryCalibratorIO.save(path, calibrator, encoder_id="encoder-v1")
    loaded = BinaryCalibratorIO.load(path, expected_encoder_id="encoder-v1")

    assert np.allclose(loaded.thresholds, calibrator.thresholds)
    assert loaded.rotation is None


def test_load_rejects_encoder_id_mismatch(tmp_path, rng: np.random.Generator) -> None:
    calibrator = BinaryCalibratorModel.fit(rng.normal(size=(8, 48)).astype(np.float32))
    path = tmp_path / "calibrator.npz"
    BinaryCalibratorIO.save(path, calibrator, encoder_id="encoder-v1")

    with pytest.raises(ValueError, match="encoder_id mismatch"):
        BinaryCalibratorIO.load(path, expected_encoder_id="encoder-v2")


def test_load_rejects_checksum_mismatch(tmp_path, rng: np.random.Generator) -> None:
    calibrator = BinaryCalibratorModel.fit(rng.normal(size=(8, 48)).astype(np.float32))
    path = tmp_path / "calibrator.npz"
    BinaryCalibratorIO.save(path, calibrator, encoder_id="encoder-v1")
    data = np.load(path, allow_pickle=True)
    tampered = tmp_path / "calibrator-tampered.npz"
    np.savez_compressed(
        tampered,
        thresholds=data["thresholds"],
        rotation=data["rotation"],
        encoder_id=data["encoder_id"],
        checksum=np.array(["bad-checksum"], dtype=object),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        BinaryCalibratorIO.load(tampered)


def test_load_rejects_bad_rotation_shape(tmp_path, rng: np.random.Generator) -> None:
    calibrator = BinaryCalibratorModel.fit(rng.normal(size=(8, 48)).astype(np.float32))
    path = tmp_path / "calibrator-bad-rotation.npz"
    np.savez_compressed(
        path,
        thresholds=calibrator.thresholds,
        rotation=np.ones((4, 4), dtype=np.float32),
        encoder_id=np.array(["encoder-v1"]),
        checksum=np.array(["unused"], dtype=object),
    )

    with pytest.raises(ValueError, match="rotation matrix shape mismatch"):
        BinaryCalibratorIO.load(path)
