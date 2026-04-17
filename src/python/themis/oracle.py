from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from themis.causal import feature_weights_from_edges, pc_skeleton
from themis.ir import tokenize
from themis.schemas import Explanation


@dataclass(slots=True)
class OracleWeights:
    semantic: float = 0.45
    causal: float = 0.35
    temporal: float = 0.20


class OracleRanker:
    def __init__(self, documents: list[dict], weights: OracleWeights | None = None):
        self.documents = {doc["id"]: doc for doc in documents}
        self.weights = weights or OracleWeights()
        self.reverse_citations: dict[str, set[str]] = defaultdict(set)
        for doc in documents:
            for cited in doc.get("metadata", {}).get("citations", []):
                self.reverse_citations[cited].add(doc["id"])

    def causal_influence(self, doc_id: str, query_terms: set[str]) -> float:
        metadata = self.documents[doc_id]["metadata"]
        tags = {tag.lower() for tag in metadata.get("tags", [])}
        cited_by = len(self.reverse_citations.get(doc_id, set()))
        cites = len(metadata.get("citations", []))
        tag_overlap = len(query_terms & tags) / max(len(query_terms), 1)
        lexical_overlap = len(query_terms & set(tokenize(self.documents[doc_id]["text"]))) / max(len(query_terms), 1)
        return 0.18 * cited_by + 0.06 * cites + 0.22 * tag_overlap + 0.12 * lexical_overlap

    def temporal_priority(self, doc_id: str) -> float:
        year = self.documents[doc_id]["metadata"].get("year", 2024)
        return max(0.0, (2026 - year) / 10)

    def learn_query_causal_profile(
        self,
        dense_results: list[tuple[str, float]],
        query: str,
    ) -> tuple[dict[str, float], list[tuple[str, str, float]]]:
        query_terms = set(tokenize(query))
        rows = []
        for doc_id, semantic_score in dense_results:
            tokens = set(tokenize(self.documents[doc_id]["text"]))
            lexical_overlap = len(tokens & query_terms) / max(len(query_terms), 1)
            cited_by = len(self.reverse_citations.get(doc_id, set()))
            citation_count = len(self.documents[doc_id]["metadata"].get("citations", []))
            temporal = self.temporal_priority(doc_id)
            pagerank = float(self.documents[doc_id]["metadata"].get("pagerank", 0.0))
            relevance_proxy = semantic_score + lexical_overlap + temporal + 0.25 * cited_by
            rows.append([semantic_score, lexical_overlap, cited_by, citation_count, temporal, pagerank, relevance_proxy])
        feature_names = [
            "semantic_score",
            "lexical_overlap",
            "cited_by",
            "citation_count",
            "temporal_priority",
            "pagerank",
            "relevance_proxy",
        ]
        samples = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(feature_names)), dtype=np.float32)
        edges = pc_skeleton(feature_names, samples, target_name="relevance_proxy", corr_threshold=0.1, max_conditioning=1)
        weights = feature_weights_from_edges(feature_names, samples, "relevance_proxy", edges)
        default_weights = {
            "semantic_score": self.weights.semantic,
            "lexical_overlap": 0.15,
            "cited_by": 0.18,
            "citation_count": 0.07,
            "temporal_priority": self.weights.temporal,
            "pagerank": 0.10,
        }
        if not weights:
            weights = default_weights
        else:
            for key, value in default_weights.items():
                weights.setdefault(key, value * 0.5)
            total = sum(weights.values())
            weights = {key: value / total for key, value in weights.items()}
        return weights, [(edge.source, edge.target, edge.strength) for edge in edges]

    def rerank(self, dense_results: list[tuple[str, float]], query: str) -> list[tuple[str, float, Explanation]]:
        query_terms = set(tokenize(query))
        learned_weights, edges = self.learn_query_causal_profile(dense_results, query)
        reranked = []
        for doc_id, semantic_score in dense_results:
            causal = self.causal_influence(doc_id, query_terms)
            temporal = self.temporal_priority(doc_id)
            lexical_overlap = len(set(tokenize(self.documents[doc_id]["text"])) & query_terms) / max(len(query_terms), 1)
            reverse_citation_count = len(self.reverse_citations.get(doc_id, set()))
            pagerank = float(self.documents[doc_id]["metadata"].get("pagerank", 0.0))
            final_score = (
                learned_weights.get("semantic_score", self.weights.semantic) * semantic_score
                + learned_weights.get("lexical_overlap", 0.0) * lexical_overlap
                + learned_weights.get("cited_by", 0.0) * reverse_citation_count
                + learned_weights.get("citation_count", 0.0) * len(self.documents[doc_id]["metadata"].get("citations", []))
                + learned_weights.get("temporal_priority", self.weights.temporal) * temporal
                + learned_weights.get("pagerank", 0.0) * pagerank
                + self.weights.causal * causal
            )
            explanation = Explanation(
                summary=f"{doc_id} is relevant because semantic similarity is reinforced by discovered causal parents for this query.",
                factors=[
                    f"semantic={semantic_score:.3f}",
                    f"causal={causal:.3f}",
                    f"lexical_overlap={lexical_overlap:.3f}",
                    f"temporal={temporal:.3f}",
                    f"cited_by={reverse_citation_count}",
                    f"pagerank={pagerank:.3f}",
                    f"learned_weights={learned_weights}",
                    f"causal_edges={edges[:4]}",
                    f"query={query}",
                ],
            )
            reranked.append((doc_id, final_score, explanation))
        return sorted(reranked, key=lambda item: (-item[1], item[0]))

    def counterfactual(self, ranked: list[tuple[str, float, Explanation]], target_doc_id: str) -> dict[str, float | str]:
        baseline = next((score for doc_id, score, _ in ranked if doc_id == target_doc_id), 0.0)
        without_target = [row for row in ranked if row[0] != target_doc_id]
        replacement = without_target[0][1] if without_target else 0.0
        delta = max(0.0, baseline - replacement)
        pct = (delta / baseline * 100.0) if baseline else 0.0
        return {
            "doc_id": target_doc_id,
            "baseline_score": round(baseline, 4),
            "replacement_score": round(replacement, 4),
            "answer_change_pct": round(pct, 2),
        }
