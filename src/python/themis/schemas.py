from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Explanation:
    summary: str
    factors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    doc_id: str
    score: float
    rank: int
    title: str
    snippet: str
    retriever: str
    explanation: Explanation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryPayload:
    raw: str
    normalized: str
    ast: dict[str, Any]
    estimated_selectivity: float | None = None
    sql: str | None = None
    elasticsearch: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkRow:
    seed: int
    dataset: str
    model: str
    query_id: str
    latency_ms: float
    metric_name: str
    metric_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
