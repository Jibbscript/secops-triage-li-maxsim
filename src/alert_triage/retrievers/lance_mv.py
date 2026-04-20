from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from alert_triage.encoders.base import EncodedTokens
from alert_triage.storage.lance_adapter import LanceTableAdapter

from ._maxsim_ref import maxsim_cosine
from .base import Candidate, QueryBundle, reject_unsupported_filter


@dataclass
class LanceMVRetriever:
    """File-backed fp16 retriever behind the `lance-mv` contract."""

    dataset_path: Path
    id: str = "lance-mv"
    _adapter: LanceTableAdapter | None = field(default=None, init=False, repr=False)

    def index(
        self,
        alert_ids: Sequence[str],
        docs: Sequence[EncodedTokens],
        texts: Sequence[str] | None = None,
    ) -> None:
        self._adapter = LanceTableAdapter.write(self.dataset_path, alert_ids, docs, texts)

    def _adapter_or_load(self) -> LanceTableAdapter:
        if self._adapter is None:
            self._adapter = LanceTableAdapter(path=self.dataset_path)
        return self._adapter

    def search(self, query: QueryBundle, k: int = 10) -> list[Candidate]:
        if query.query_fp16 is None:
            raise ValueError("query.query_fp16 is required for LanceMVRetriever")
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
            for alert_id, _text, doc_fp16 in self._adapter_or_load().iter_documents()
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.alert_id))
        return hits[:k]

    def size(self) -> int:
        return self._adapter_or_load().size()
