from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class IndexedDocument:
    doc_id: str
    title: str
    text: str
    metadata: dict
    tokens: list[str]


class ClassicalIndex:
    def __init__(self, raw_documents: Iterable[dict]):
        self.documents: dict[str, IndexedDocument] = {}
        self.inverted_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.positions: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self.doc_lengths: dict[str, int] = {}
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: dict[str, Counter[str]] = {}
        self.citation_graph: dict[str, set[str]] = defaultdict(set)
        self.reverse_citations: dict[str, set[str]] = defaultdict(set)

        for doc in raw_documents:
            tokens = tokenize(f"{doc['title']} {doc['text']}")
            indexed = IndexedDocument(
                doc_id=doc["id"],
                title=doc["title"],
                text=doc["text"],
                metadata=doc["metadata"],
                tokens=tokens,
            )
            self.documents[indexed.doc_id] = indexed
            self.doc_lengths[indexed.doc_id] = len(tokens)
            tf = Counter(tokens)
            self.term_freqs[indexed.doc_id] = tf
            for term, freq in tf.items():
                self.inverted_index[term].append((indexed.doc_id, freq))
                self.doc_freq[term] += 1
            for pos, term in enumerate(tokens):
                self.positions[term][indexed.doc_id].append(pos)
            citations = set(doc.get("metadata", {}).get("citations", []))
            self.citation_graph[indexed.doc_id] = citations
            for cited in citations:
                self.reverse_citations[cited].add(indexed.doc_id)

        self.avg_doc_len = sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)
        self.page_rank = self._compute_pagerank()

    def term_lookup(self, term: str) -> list[tuple[str, int]]:
        return list(self.inverted_index.get(term.lower(), []))

    def phrase_query(self, phrase: str) -> list[str]:
        phrase_terms = tokenize(phrase)
        if not phrase_terms:
            return []
        candidate_docs = set(self.positions.get(phrase_terms[0], {}).keys())
        for term in phrase_terms[1:]:
            candidate_docs &= set(self.positions.get(term, {}).keys())
        matched: list[str] = []
        for doc_id in candidate_docs:
            first_positions = self.positions[phrase_terms[0]][doc_id]
            for start in first_positions:
                if all((start + offset) in self.positions[term][doc_id] for offset, term in enumerate(phrase_terms)):
                    matched.append(doc_id)
                    break
        return sorted(matched)

    def boolean_and(self, terms: list[str]) -> list[str]:
        doc_sets = [set(doc_id for doc_id, _ in self.term_lookup(term)) for term in terms]
        return sorted(set.intersection(*doc_sets)) if doc_sets else []

    def boolean_or(self, terms: list[str]) -> list[str]:
        matches: set[str] = set()
        for term in terms:
            matches.update(doc_id for doc_id, _ in self.term_lookup(term))
        return sorted(matches)

    def boolean_not(self, term: str) -> list[str]:
        matched = {doc_id for doc_id, _ in self.term_lookup(term)}
        return sorted(set(self.documents) - matched)

    def near_query(self, left: str, right: str, distance: int) -> list[str]:
        left_docs = set(self.positions.get(left.lower(), {}).keys())
        right_docs = set(self.positions.get(right.lower(), {}).keys())
        matched = []
        for doc_id in sorted(left_docs & right_docs):
            left_positions = self.positions[left.lower()][doc_id]
            right_positions = self.positions[right.lower()][doc_id]
            if any(abs(lp - rp) <= distance for lp in left_positions for rp in right_positions):
                matched.append(doc_id)
        return matched

    def evaluate_query_ast(self, ast: dict[str, Any]) -> set[str]:
        ast_type = ast.get("type")
        if ast_type == "Term":
            return {doc_id for doc_id, _ in self.term_lookup(ast["value"])}
        if ast_type == "Phrase":
            return set(self.phrase_query(" ".join(ast["terms"])))
        if ast_type == "Near":
            return set(self.near_query(ast["left"], ast["right"], int(ast["distance"])))
        if ast_type == "Boost":
            return self.evaluate_query_ast(ast["query"])
        if ast_type == "And":
            return self.evaluate_query_ast(ast["left"]) & self.evaluate_query_ast(ast["right"])
        if ast_type == "Or":
            return self.evaluate_query_ast(ast["left"]) | self.evaluate_query_ast(ast["right"])
        if ast_type == "Not":
            return set(self.documents) - self.evaluate_query_ast(ast["query"])
        return set(self.documents)

    def tfidf_scores(self, query: str) -> dict[str, float]:
        scores = defaultdict(float)
        terms = tokenize(query)
        num_docs = len(self.documents)
        for term in terms:
            idf = math.log((num_docs + 1) / (1 + self.doc_freq.get(term, 0))) + 1
            for doc_id, tf in self.term_lookup(term):
                scores[doc_id] += tf * idf
        return dict(scores)

    def bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        scores = defaultdict(float)
        terms = tokenize(query)
        num_docs = len(self.documents)
        for term in terms:
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in self.term_lookup(term):
                dl = self.doc_lengths[doc_id]
                denom = tf + k1 * (1 - b + b * dl / self.avg_doc_len)
                scores[doc_id] += idf * ((tf * (k1 + 1)) / denom)
        return dict(scores)

    def dirichlet_scores(self, query: str, mu: float = 1200.0) -> dict[str, float]:
        scores = defaultdict(float)
        collection_len = sum(self.doc_lengths.values())
        collection_counts = Counter()
        for tf in self.term_freqs.values():
            collection_counts.update(tf)
        query_terms = tokenize(query)
        for doc_id, tf in self.term_freqs.items():
            dl = self.doc_lengths[doc_id]
            score = 0.0
            for term in query_terms:
                p_wc = collection_counts.get(term, 0) / max(collection_len, 1)
                score += math.log((tf.get(term, 0) + mu * p_wc + 1e-9) / (dl + mu))
            scores[doc_id] = score
        return dict(scores)

    def pagerank_scores(self) -> dict[str, float]:
        return dict(self.page_rank)

    def rank(self, query: str, model: str, candidate_doc_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if model == "tfidf":
            scores = self.tfidf_scores(query)
        elif model == "bm25":
            scores = self.bm25_scores(query)
        elif model == "lm":
            scores = self.dirichlet_scores(query)
        elif model == "pagerank":
            scores = self.pagerank_scores()
        else:
            raise ValueError(f"Unknown ranking model: {model}")
        if candidate_doc_ids is not None:
            scores = {doc_id: score for doc_id, score in scores.items() if doc_id in candidate_doc_ids}
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def _compute_pagerank(self, damping: float = 0.85, iterations: int = 20) -> dict[str, float]:
        doc_ids = list(self.documents)
        if not doc_ids:
            return {}
        num_docs = len(doc_ids)
        scores = {doc_id: 1.0 / num_docs for doc_id in doc_ids}
        for _ in range(iterations):
            new_scores = {doc_id: (1 - damping) / num_docs for doc_id in doc_ids}
            for doc_id in doc_ids:
                outlinks = self.citation_graph.get(doc_id) or set()
                if outlinks:
                    share = damping * scores[doc_id] / len(outlinks)
                    for target in outlinks:
                        if target in new_scores:
                            new_scores[target] += share
                else:
                    share = damping * scores[doc_id] / num_docs
                    for target in doc_ids:
                        new_scores[target] += share
            scores = new_scores
        for doc_id, doc in self.documents.items():
            scores[doc_id] += float(doc.metadata.get("pagerank", 0.0))
        return scores
