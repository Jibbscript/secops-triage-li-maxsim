from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from alert_triage.encoders.base import EncodedTokens


FilterScalar = str | int | float | bool
FilterValue = FilterScalar | tuple[FilterScalar, ...]
FilterOperator = Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in"]


class FilterClause(BaseModel):
    """Typed filter shell reserved for later lowering work."""

    model_config = ConfigDict(extra="forbid")

    field: str
    op: FilterOperator
    value: FilterValue


class QueryFilter(BaseModel):
    """Placeholder structured filter contract.

    Phase-1 keeps this as a typed boundary only. Later phases will lower these
    clauses into storage-specific execution paths.
    """

    model_config = ConfigDict(extra="forbid")

    clauses: tuple[FilterClause, ...] = ()
    combinator: Literal["and"] = "and"

    def is_noop(self) -> bool:
        return len(self.clauses) == 0


class QueryBundle(BaseModel):
    """Canonical retrieval input across lexical, fp16, binary, and staged retrievers."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    alert_id: str | None = None
    query_text: str | None = None
    query_fp16: np.ndarray | None = None
    query_bin: np.ndarray | None = None
    filter: QueryFilter | None = None
    filter_expr: str | None = None

    @model_validator(mode="after")
    def reject_raw_filter_expr(self) -> "QueryBundle":
        if self.filter_expr is not None:
            raise ValueError(
                "Raw SQL filter strings are not supported in phase-1; use QueryBundle.filter."
            )
        return self


StageName = Literal["bm25", "binary_hamming", "fp16_maxsim", "hybrid"]


@dataclass(frozen=True)
class Candidate:
    """One candidate retrieved from a search stage."""

    alert_id: str
    score: float
    stage: StageName
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


def reject_unsupported_filter(query: QueryBundle) -> None:
    """Guard the typed filter boundary until lowering exists."""

    if query.filter is not None and not query.filter.is_noop():
        raise NotImplementedError("Structured filters are not supported in phase-1.")


def lower_structured_filter(query_filter: QueryFilter | None):
    """Lower an allowlisted structured filter into a DataFusion expression."""

    if query_filter is None or query_filter.is_noop():
        return None

    from datafusion import col, lit

    def lower_clause(clause: FilterClause):
        field = col(clause.field)
        value = clause.value

        if clause.op in {"in", "not_in"}:
            if not isinstance(value, tuple) or len(value) == 0:
                raise ValueError(f"{clause.op} filters require a non-empty tuple value")
            comparisons = [(field == lit(item)) for item in value]
            if clause.op == "in":
                return reduce(lambda left, right: left | right, comparisons)
            return reduce(lambda left, right: left & right, ((field != lit(item)) for item in value))

        if isinstance(value, tuple):
            raise ValueError(f"{clause.op} filters do not accept tuple values")
        if clause.op == "eq":
            return field == lit(value)
        if clause.op == "ne":
            return field != lit(value)
        if clause.op == "lt":
            return field < lit(value)
        if clause.op == "lte":
            return field <= lit(value)
        if clause.op == "gt":
            return field > lit(value)
        if clause.op == "gte":
            return field >= lit(value)
        raise ValueError(f"unsupported filter operator: {clause.op}")

    lowered = [lower_clause(clause) for clause in query_filter.clauses]
    return reduce(lambda left, right: left & right, lowered)
