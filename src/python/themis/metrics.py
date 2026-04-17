from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc in retrieved[:k] if doc in relevant) / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p_at_k = precision_at_k(retrieved, relevant, k)
    r_at_k = recall_at_k(retrieved, relevant, k)
    if p_at_k + r_at_k == 0:
        return 0.0
    return 2 * p_at_k * r_at_k / (p_at_k + r_at_k)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    running = 0.0
    hits = 0
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            running += hits / idx
    return running / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    dcg = 0.0
    for idx, doc_id in enumerate(top, start=1):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg
