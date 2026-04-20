from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from alert_triage.encoders.base import EncodedTokens
from alert_triage.storage.in_memory import InMemoryTokenVectorStore

from ._maxsim_ref import maxsim_cosine
from .base import Candidate, QueryBundle, reject_unsupported_filter


@dataclass
class FP16ReferenceRetriever:
    """Python-native fp16 reference retriever for the phase-1 harness."""

    vector_store: InMemoryTokenVectorStore = field(default_factory=InMemoryTokenVectorStore)
    id: str = "fp16-ref"

    def index(
        self,
        alert_ids: Sequence[str],
        docs: Sequence[EncodedTokens],
        texts=None,
    ) -> None:
        self.vector_store.upsert(alert_ids, docs)

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        if query.query_fp16 is None:
            raise ValueError("query.query_fp16 is required for FP16ReferenceRetriever")
        if k < 1:
            raise ValueError("k must be >= 1")
        reject_unsupported_filter(query)

        hits = [
            Candidate(
                alert_id=alert_id,
                score=float(maxsim_cosine(query.query_fp16, doc_fp16)),
                stage="fp16_maxsim",
                debug={},
            )
            for alert_id, doc_fp16 in self.vector_store.iter_fp16()
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.alert_id))
        return hits[:k]

    def size(self) -> int:
        return self.vector_store.size()
