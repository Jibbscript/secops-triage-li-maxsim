from __future__ import annotations

from dataclasses import dataclass

from ._maxsim_ref import maxsim_cosine
from .base import Candidate, QueryBundle, Retriever, TokenVectorStore, reject_unsupported_filter


@dataclass
class BinaryThenFP16RerankRetriever:
    """Two-stage retrieval: binary Hamming candidate generation, then fp16 rerank."""

    candidate_retriever: Retriever
    vector_store: TokenVectorStore
    prefilter_top_n: int = 200
    id: str = "binary-then-fp16-rerank"

    def index(self, alert_ids, docs, texts=None) -> None:
        # Delegated to the underlying stores and candidate retriever.
        return None

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        if query.query_bin is None:
            raise ValueError("query.query_bin is required")
        if query.query_fp16 is None:
            raise ValueError("query.query_fp16 is required")
        if k < 1:
            raise ValueError("k must be >= 1")
        reject_unsupported_filter(query)

        coarse = self.candidate_retriever.search(query, k=self.prefilter_top_n)
        if not coarse:
            return []

        doc_map = self.vector_store.fetch_fp16([c.alert_id for c in coarse])

        rescored: list[Candidate] = []
        for c in coarse:
            doc = doc_map.get(c.alert_id)
            if doc is None:
                continue
            rerank_score = float(maxsim_cosine(query.query_fp16, doc))
            rescored.append(
                Candidate(
                    alert_id=c.alert_id,
                    score=rerank_score,
                    stage="fp16_maxsim",
                    debug={"binary_score": c.score},
                )
            )

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:k]

    def size(self) -> int:
        return self.candidate_retriever.size()
