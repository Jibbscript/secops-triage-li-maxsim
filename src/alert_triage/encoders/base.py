from __future__ import annotations

from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


QuantizeMode = Literal["fp16", "binary", "both"]
BINARY_BITS = 48


class EncodedTokens(BaseModel):
    """Container for one late-interaction encoding result.

    Shapes:
    - fp16: (tokens, dim) float16|float32
    - binary: (tokens, 8) uint8 for a 48-bit representation padded to 8 bytes
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    fp16: np.ndarray | None = None
    binary: np.ndarray | None = None

    def require_fp16(self) -> np.ndarray:
        if self.fp16 is None:
            raise ValueError("fp16 encoding is not available")
        return self.fp16

    def require_binary(self) -> np.ndarray:
        if self.binary is None:
            raise ValueError("binary encoding is not available")
        return self.binary


class BinaryCalibratorModel(BaseModel):
    """Concrete serializable calibrator payload used by binary quantization."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    thresholds: np.ndarray = Field(..., description="shape=(48,), dtype=float32")
    rotation: np.ndarray | None = Field(
        default=None,
        description="optional orthogonal matrix, shape=(48,48), dtype=float32",
    )

    @classmethod
    def fit(cls, float_vectors: np.ndarray) -> "BinaryCalibratorModel":
        """Fit a simple per-dimension median-threshold calibrator.

        Parameters
        ----------
        float_vectors:
            Array of shape (n_tokens, dim). Expected dim=48 for mxbai-edge-colbert-v0-17m.
        """
        if float_vectors.ndim != 2:
            raise ValueError(f"expected 2D array, got shape={float_vectors.shape}")
        if float_vectors.shape[1] != BINARY_BITS:
            raise ValueError(
                f"expected {BINARY_BITS}-dim token vectors, got shape={float_vectors.shape}"
            )
        thresholds = np.median(float_vectors, axis=0).astype(np.float32, copy=False)
        return cls(thresholds=thresholds, rotation=None)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Convert float token vectors into padded binary token rows.

        Returns
        -------
        np.ndarray
            Array of shape (tokens, 8), dtype=uint8. Only the low 48 bits are meaningful.
        """
        if x.ndim != 2:
            raise ValueError(f"expected 2D array, got shape={x.shape}")
        if x.shape[1] != BINARY_BITS:
            raise ValueError(
                f"expected {BINARY_BITS}-dim token vectors, got shape={x.shape}"
            )
        if x.shape[1] != self.thresholds.shape[0]:
            raise ValueError(
                f"dim mismatch: x.shape[1]={x.shape[1]} thresholds={self.thresholds.shape[0]}"
            )

        y = x @ self.rotation if self.rotation is not None else x
        bits = (y > self.thresholds).astype(np.uint8, copy=False)
        packed = np.packbits(bits, axis=-1, bitorder="little")
        if packed.shape[1] != 6:
            raise ValueError(
                f"expected 6 packed bytes for {BINARY_BITS} bits, got shape={packed.shape}"
            )
        out = np.zeros((packed.shape[0], 8), dtype=np.uint8)
        out[:, : packed.shape[1]] = packed
        return out


@runtime_checkable
class BinaryCalibrator(Protocol):
    thresholds: np.ndarray
    rotation: np.ndarray | None

    @classmethod
    def fit(cls, float_vectors: np.ndarray) -> "BinaryCalibrator":
        ...

    def apply(self, x: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class LateInteractionEncoder(Protocol):
    """Late-interaction encoder with optional binary quantization.

    Implementations must be deterministic for fixed weights and input text.
    """

    id: str
    dim: int
    binary_stride: int

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        quantize: QuantizeMode = "fp16",
        calibrator: BinaryCalibrator | None = None,
    ) -> list[EncodedTokens]:
        """Encode query texts.

        Returns a list of EncodedTokens, one per query.
        """
        ...

    def encode_docs(
        self,
        texts: Sequence[str],
        *,
        quantize: QuantizeMode = "fp16",
        calibrator: BinaryCalibrator | None = None,
    ) -> list[EncodedTokens]:
        """Encode document texts.

        Returns a list of EncodedTokens, one per document.
        """
        ...
