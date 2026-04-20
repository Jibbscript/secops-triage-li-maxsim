from __future__ import annotations

from dataclasses import dataclass

import time

from .base import Candidate, QueryBundle, Retriever, TokenVectorStore
from .binary_rerank import rerank_fp16_candidates


@dataclass
class HybridBM25ThenFP16RerankRetriever:
    """Two-stage retrieval: lexical candidate generation, then fp16 rerank."""

    candidate_retriever: Retriever
    vector_store: TokenVectorStore
    prefilter_top_n: int = 200
    id: str = "hybrid-bm25-then-fp16-rerank"

    def index(self, alert_ids, docs, texts=None) -> None:
        # Delegated to the underlying candidate retriever and vector store.
        return None

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        hits, _timings = self.search_with_timings(query, k=k)
        return hits

    def search_with_timings(self, query: QueryBundle, k: int = 10) -> tuple[list[Candidate], dict[str, float]]:
        if query.query_text is None:
            raise ValueError("query.query_text is required")
        if query.query_fp16 is None:
            raise ValueError("query.query_fp16 is required")
        if k < 1:
            raise ValueError("k must be >= 1")

        candidate_started = time.perf_counter()
        coarse = self.candidate_retriever.search(query, k=self.prefilter_top_n)
        candidate_ms = (time.perf_counter() - candidate_started) * 1000.0
        if not coarse:
            return [], {"candidate_ms": candidate_ms, "rerank_ms": 0.0}

        rerank_started = time.perf_counter()
        hits = rerank_fp16_candidates(
            coarse=coarse,
            vector_store=self.vector_store,
            query_fp16=query.query_fp16,
            k=k,
            stage="hybrid",
            debug_score_key="bm25_score",
        )
        rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
        return hits, {"candidate_ms": candidate_ms, "rerank_ms": rerank_ms}

    def size(self) -> int:
        return self.candidate_retriever.size()
