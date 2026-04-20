from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from alert_triage.encoders.base import EncodedTokens


def _doc_to_nested_list(doc: np.ndarray) -> list[list[float]]:
    arr = np.asarray(doc, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D token matrix, got shape={arr.shape}")
    return [[float(value) for value in row] for row in arr]


def _doc_from_cell(value) -> np.ndarray:
    if hasattr(value, "as_py"):
        value = value.as_py()
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class LanceTableAdapter:
    """Arrow-backed stand-in for the future Lance multivector corpus surface.

    This keeps the `lance-mv` retrieval contract executable without claiming the
    upstream Lance package is installed in the Phase-3 slice.
    """

    path: Path

    @property
    def table_path(self) -> Path:
        return self.path / "multivectors.arrow"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @classmethod
    def write(
        cls,
        path: Path,
        alert_ids: Sequence[str],
        docs: Sequence[EncodedTokens],
        texts: Sequence[str] | None,
    ) -> "LanceTableAdapter":
        if texts is None:
            raise ValueError("texts are required for LanceTableAdapter.write")
        if len(alert_ids) != len(docs) or len(alert_ids) != len(texts):
            raise ValueError("alert_ids, docs, and texts must have matching lengths")

        path.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "alert_id": pa.array([str(alert_id) for alert_id in alert_ids], type=pa.string()),
                "text": pa.array([str(text) for text in texts], type=pa.string()),
                "mv_fp16": pa.array(
                    [_doc_to_nested_list(doc.require_fp16()) for doc in docs],
                    type=pa.list_(pa.list_(pa.float32())),
                ),
            }
        )
        with pa.OSFile(str(path / "multivectors.arrow"), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write(table)

        manifest = {
            "backend": "lance-adapter",
            "format": "arrow-ipc",
            "row_count": len(alert_ids),
            "note": (
                "Phase-3 executable adapter for the lance-mv surface. "
                "This is not the upstream Lance package or storage engine."
            ),
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return cls(path=path)

    def _read_table(self) -> pa.Table:
        with pa.memory_map(str(self.table_path), "r") as source:
            return ipc.RecordBatchFileReader(source).read_all()

    def fetch_fp16(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        wanted = set(ids)
        table = self._read_table()
        return {
            str(alert_id.as_py()): _doc_from_cell(doc)
            for alert_id, doc in zip(table["alert_id"], table["mv_fp16"], strict=True)
            if str(alert_id.as_py()) in wanted
        }

    def iter_documents(self) -> list[tuple[str, str, np.ndarray]]:
        table = self._read_table()
        return [
            (str(alert_id.as_py()), str(text.as_py()), _doc_from_cell(doc))
            for alert_id, text, doc in zip(
                table["alert_id"], table["text"], table["mv_fp16"], strict=True
            )
        ]

    def size(self) -> int:
        return len(self._read_table())
