from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from alert_triage.encoders.base import EncodedTokens

from .base import Candidate, QueryBundle, reject_unsupported_filter


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class BM25Retriever:
    """Small deterministic lexical retriever for the checked-in Phase-3 fixtures."""

    k1: float = 1.2
    b: float = 0.75
    id: str = "bm25"
    _term_freqs: dict[str, Counter[str]] = field(default_factory=dict, init=False)
    _doc_freqs: Counter[str] = field(default_factory=Counter, init=False)
    _doc_lengths: dict[str, int] = field(default_factory=dict, init=False)
    _avg_doc_length: float = field(default=0.0, init=False)

    def index(
        self,
        alert_ids: Sequence[str],
        docs: Sequence[EncodedTokens],
        texts: Sequence[str] | None = None,
    ) -> None:
        del docs  # lexical retrieval is text-only for the Phase-3 fixture slice
        if texts is None:
            raise ValueError("texts are required for BM25Retriever.index")
        if len(alert_ids) != len(texts):
            raise ValueError("alert_ids and texts must have matching lengths")

        self._term_freqs.clear()
        self._doc_freqs.clear()
        self._doc_lengths.clear()

        total_terms = 0
        for alert_id, text in zip(alert_ids, texts, strict=True):
            tokens = _tokenize(text)
            term_freq = Counter(tokens)
            self._term_freqs[alert_id] = term_freq
            self._doc_lengths[alert_id] = len(tokens)
            total_terms += len(tokens)
            for token in term_freq:
                self._doc_freqs[token] += 1

        self._avg_doc_length = total_terms / len(alert_ids) if alert_ids else 0.0

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        if query.query_text is None:
            raise ValueError("query.query_text is required for BM25Retriever")
        if k < 1:
            raise ValueError("k must be >= 1")
        reject_unsupported_filter(query)

        query_terms = _tokenize(query.query_text)
        if not query_terms:
            return []

        num_docs = len(self._term_freqs)
        if num_docs == 0:
            return []

        hits: list[Candidate] = []
        for alert_id in sorted(self._term_freqs):
            term_freqs = self._term_freqs[alert_id]
            doc_length = self._doc_lengths[alert_id]
            score = 0.0
            matched_terms: list[str] = []
            for term in query_terms:
                tf = term_freqs.get(term, 0)
                if tf == 0:
                    continue
                matched_terms.append(term)
                doc_freq = self._doc_freqs[term]
                idf = math.log(1.0 + ((num_docs - doc_freq + 0.5) / (doc_freq + 0.5)))
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * doc_length / max(self._avg_doc_length, 1.0)
                )
                score += idf * ((tf * (self.k1 + 1.0)) / denom)
            if score > 0.0:
                hits.append(
                    Candidate(
                        alert_id=alert_id,
                        score=float(score),
                        stage="bm25",
                        debug={"matched_terms": float(len(set(matched_terms)))},
                    )
                )

        hits.sort(key=lambda hit: (-hit.score, hit.alert_id))
        return hits[:k]

    def size(self) -> int:
        return len(self._term_freqs)
