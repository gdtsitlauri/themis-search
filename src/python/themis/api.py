from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel

from themis.dataset import load_documents
from themis.haskell_bridge import parse_query
from themis.ir import ClassicalIndex
from themis.neural import CrossEncoderReranker, DenseRetriever, hybrid_fusion
from themis.oracle import OracleRanker
from themis.query_language import parse_query_text, payload_from_ast
from themis.schemas import QueryPayload, SearchResult


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


documents = load_documents()
classical_index = ClassicalIndex(documents)
dense = DenseRetriever(documents, prefer_gpu=True)
cross = CrossEncoderReranker(documents, prefer_gpu=True)
oracle = OracleRanker(documents)
app = FastAPI(title="THEMIS API", version="0.1.0")


def _parse_request_query(query: str):
    try:
        ast = parse_query_text(query)
        return QueryPayload(**payload_from_ast(query, ast))
    except Exception:
        return parse_query(query)


def _warm_runtime() -> None:
    try:
        warm_query = "causal relevance search"
        dense_ranked = dense.search(warm_query, top_k=min(3, len(documents)))
        if dense_ranked:
            cross.rerank(warm_query, dense_ranked)
    except Exception:
        pass


_warm_runtime()


@app.get("/index")
def get_index_stats() -> dict:
    return {
        "documents": len(classical_index.documents),
        "terms": len(classical_index.inverted_index),
        "avg_doc_len": round(classical_index.avg_doc_len, 3),
        "dense_index": dense.index_stats(),
        "runtime_device": dense.index_stats()["device"],
    }


@app.post("/search")
def search(request: SearchRequest) -> dict:
    started = time.perf_counter()
    parsed = _parse_request_query(request.query)
    candidate_ids = classical_index.evaluate_query_ast(parsed.ast)
    query_text = parsed.normalized or request.query
    if not candidate_ids:
        candidate_ids = set(classical_index.documents)
    sparse = classical_index.bm25_scores(query_text)
    if candidate_ids != set(classical_index.documents):
        sparse = {doc_id: score for doc_id, score in sparse.items() if doc_id in candidate_ids}
    dense_ranked = dense.search(query_text, top_k=max(request.top_k * 2, 5), approximate=len(classical_index.documents) >= 64)
    if candidate_ids != set(classical_index.documents):
        dense_ranked = [item for item in dense_ranked if item[0] in candidate_ids]
    reranked = cross.rerank(query_text, dense_ranked)
    fused = hybrid_fusion(sparse, reranked, alpha=0.55)[: max(request.top_k, 1)]
    oracle_ranked = oracle.rerank(fused, request.query)
    results = []
    for rank, (doc_id, score, explanation) in enumerate(oracle_ranked[: request.top_k], start=1):
        doc = classical_index.documents[doc_id]
        results.append(
            SearchResult(
                doc_id=doc_id,
                score=round(score, 4),
                rank=rank,
                title=doc.title,
                snippet=doc.text[:120],
                retriever="themis_oracle",
                explanation=explanation,
            ).to_dict()
        )
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "query": parsed.to_dict(),
        "results": results,
        "timing_ms": round(elapsed, 3),
        "candidate_count": len(candidate_ids),
    }
