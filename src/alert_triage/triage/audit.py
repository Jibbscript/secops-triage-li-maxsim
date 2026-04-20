from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict


AuditKind = Literal["model_call", "tool_call"]


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: AuditKind
    name: str
    inputs: dict[str, object]
    outputs: dict[str, object]


class TraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orphan_llm_calls: int
    orphan_tool_calls: int


def summarize_audit(
    records_by_sample: Sequence[Sequence[AuditRecord]],
    *,
    expected_llm_names: set[str],
    expected_llm_roles: set[str] | None = None,
    expected_tool_names: set[str] | None = None,
) -> TraceSummary:
    tool_expectations = expected_tool_names or {"propose_investigation_step"}
    missing_llm = 0
    missing_tool = 0
    for records in records_by_sample:
        llm_names = {record.name for record in records if record.kind == "model_call"}
        llm_roles = {
            str(record.inputs.get("role"))
            for record in records
            if record.kind == "model_call" and isinstance(record.inputs.get("role"), str)
        }
        tool_names = {record.name for record in records if record.kind == "tool_call"}
        missing_llm += len(expected_llm_names - llm_names)
        if expected_llm_roles is not None:
            missing_llm += len(expected_llm_roles - llm_roles)
        missing_tool += len(tool_expectations - tool_names)
    return TraceSummary(
        orphan_llm_calls=missing_llm,
        orphan_tool_calls=missing_tool,
    )


def write_trace_jsonl(path: Path, samples: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.model_dump(mode="json"), sort_keys=True) + "\n")
