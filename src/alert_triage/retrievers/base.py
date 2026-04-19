from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from alert_triage.encoders.base import EncodedTokens


class QueryBundle(BaseModel):
    """Canonical retrieval input across lexical, fp16, binary, and staged retrievers."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    alert_id: str | None = None
    query_text: str | None = None
    query_fp16: np.ndarray | None = None
    query_bin: np.ndarray | None = None
    filter_expr: str | None = None


@dataclass(frozen=True)
class Candidate:
    """One candidate retrieved from a search stage."""

    alert_id: str
    score: float
    stage: str
    debug: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class TokenVectorStore(Protocol):
    """Protocol for fetching stored multivectors for rerank or hydration."""

    def fetch_fp16(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        ...

    def fetch_bin(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        ...


@runtime_checkable
class Retriever(Protocol):
    """Unified retriever contract.

    Notes
    -----
    - lexical retrievers consume query_text
    - fp16 retrievers consume query_fp16
    - binary retrievers consume query_bin
    - staged retrievers may consume more than one field
    """

    id: str

    def index(
        self,
        alert_ids: Sequence[str],
        docs: Sequence[EncodedTokens],
        texts: Sequence[str] | None = None,
    ) -> None:
        ...

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        ...

    def size(self) -> int:
        ...
