from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alert_triage.encoders.base import EncodedTokens
from alert_triage.retrievers.base import QueryBundle
from alert_triage.retrievers.lance_mv import LanceMVRetriever


def test_lance_mv_retriever_round_trips_file_backed_fixture_store(tmp_path: Path) -> None:
    retriever = LanceMVRetriever(dataset_path=tmp_path / "fixture.lance")
    retriever.index(
        ["alert-1", "alert-2", "alert-3"],
        [
            EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)),
            EncodedTokens(fp16=np.asarray([[0.0, 1.0]], dtype=np.float32)),
            EncodedTokens(fp16=np.asarray([[1.0, 1.0]], dtype=np.float32)),
        ],
        texts=["phishing", "ransomware", "mixed"],
    )

    hits = retriever.search(QueryBundle(query_fp16=np.asarray([[1.0, 0.0]], dtype=np.float32)), k=2)

    assert [hit.alert_id for hit in hits] == ["alert-1", "alert-3"]
    assert retriever.size() == 3
    manifest = (tmp_path / "fixture.lance" / "manifest.json").read_text()
    assert "lance-adapter" in manifest


def test_lance_mv_retriever_requires_fp16_query(tmp_path: Path) -> None:
    retriever = LanceMVRetriever(dataset_path=tmp_path / "fixture.lance")
    retriever.index(
        ["alert-1"],
        [EncodedTokens(fp16=np.asarray([[1.0, 0.0]], dtype=np.float32))],
        texts=["phishing"],
    )

    with pytest.raises(ValueError, match="query.query_fp16 is required for LanceMVRetriever"):
        retriever.search(QueryBundle(query_text="phishing"))
