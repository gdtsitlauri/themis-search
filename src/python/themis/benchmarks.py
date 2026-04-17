from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from themis.dataset import (
    external_resource_status,
    expand_documents,
    generate_synthetic_collection,
    load_fixture,
    load_optional_forensics_collection,
    load_optional_ir_collection,
    results_dir,
)
from themis.forensics import ForensicsAnalyzer
from themis.ir import ClassicalIndex
from themis.metrics import average_precision, f1_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from themis.neural import CrossEncoderReranker, DenseRetriever, LateInteractionRetriever, hybrid_fusion, tune_alpha
from themis.oracle import OracleRanker


SEEDS = (7, 11, 19)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ranked_doc_ids(pairs: list[tuple[str, float]]) -> list[str]:
    return [doc_id for doc_id, _ in pairs]


def run_ir_benchmarks() -> list[dict]:
    rows: list[dict] = []
    comparison_rows: list[dict] = []
    for seed in SEEDS:
        docs, queries = generate_synthetic_collection(seed, num_docs=180, num_queries=30)
        index = ClassicalIndex(docs)
        for model in ("tfidf", "bm25", "lm", "pagerank"):
            aggregate_map = []
            for query in queries:
                start = time.perf_counter()
                ranked = index.rank(query["text"], model)
                latency_ms = (time.perf_counter() - start) * 1000
                relevant = set(query["relevant"])
                retrieved = _ranked_doc_ids(ranked)
                metric_values = {
                    "precision@3": precision_at_k(retrieved, relevant, 3),
                    "recall@3": recall_at_k(retrieved, relevant, 3),
                    "f1@3": f1_at_k(retrieved, relevant, 3),
                    "map": average_precision(retrieved, relevant),
                    "ndcg@3": ndcg_at_k(retrieved, relevant, 3),
                    "mrr": reciprocal_rank(retrieved, relevant),
                }
                aggregate_map.append(metric_values["map"])
                for name, value in metric_values.items():
                    rows.append(
                        {
                            "seed": seed,
                            "dataset": "synthetic_research_collection",
                            "model": model,
                            "query_id": query["id"],
                            "latency_ms": round(latency_ms, 3),
                            "metric_name": name,
                            "metric_value": round(value, 4),
                        }
                    )
            comparison_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "mean_map": round(sum(aggregate_map) / len(aggregate_map), 4),
                    "queries": len(queries),
                }
            )
    out_root = results_dir() / "ir"
    _write_csv(out_root / "retrieval_benchmarks.csv", rows, list(rows[0].keys()))
    _write_csv(out_root / "ranking_comparison.csv", comparison_rows, list(comparison_rows[0].keys()))
    return rows


def run_neural_benchmarks() -> list[dict]:
    rows: list[dict] = []
    latency_rows: list[dict] = []
    base_docs, _ = generate_synthetic_collection(SEEDS[0], num_docs=120, num_queries=12)
    for seed in SEEDS:
        docs, queries = generate_synthetic_collection(seed, num_docs=150, num_queries=24)
        index = ClassicalIndex(docs)
        dense = DenseRetriever(docs, prefer_gpu=True)
        cross = CrossEncoderReranker(docs, prefer_gpu=True)
        late = LateInteractionRetriever(docs, prefer_gpu=True)
        validation_rows = [
            (index.bm25_scores(query["text"]), dense.search(query["text"]), set(query["relevant"])) for query in queries[:8]
        ]
        alpha = tune_alpha(validation_rows)
        for query in queries:
            start = time.perf_counter()
            dense_ranked = dense.search(query["text"])
            reranked = cross.rerank(query["text"], dense_ranked)
            late_ranked = late.search(query["text"])
            hybrid_ranked = hybrid_fusion(index.bm25_scores(query["text"]), reranked, alpha)
            latency_ms = (time.perf_counter() - start) * 1000
            relevant = set(query["relevant"])
            for model, ranked in (
                ("dense_exact", dense_ranked),
                ("cross_encoder", reranked),
                ("late_interaction", late_ranked),
                ("hybrid", hybrid_ranked),
            ):
                doc_ids = _ranked_doc_ids(ranked)
                rows.append(
                    {
                        "seed": seed,
                        "dataset": "synthetic_research_collection",
                        "model": model,
                        "query_id": query["id"],
                        "latency_ms": round(latency_ms, 3),
                        "metric_name": "ndcg@3",
                        "metric_value": round(ndcg_at_k(doc_ids, relevant, 3), 4),
                        "backend": dense.index_stats()["backend"] if model.startswith("dense") else getattr(cross, "backend", "heuristic"),
                    }
                )
        for size in (100, 300, 600, 1000):
            latency_docs = expand_documents(base_docs, size)
            dense_latency = DenseRetriever(latency_docs, prefer_gpu=True)
            start = time.perf_counter()
            dense_latency.search("causal relevance search", top_k=10, approximate=size >= 300)
            latency_ms = (time.perf_counter() - start) * 1000
            latency_rows.append(
                {
                    "seed": seed,
                    "index_size": size,
                    "query": "causal relevance search",
                    "dense_latency_ms": round(latency_ms, 3),
                    "approximate": bool(size >= 300),
                    "backend": dense_latency.index_stats()["backend"],
                    "device": dense_latency.index_stats()["device"],
                }
            )
    out_root = results_dir() / "neural_search"
    _write_csv(out_root / "dense_vs_sparse.csv", rows, list(rows[0].keys()))
    _write_csv(out_root / "latency_benchmark.csv", latency_rows, list(latency_rows[0].keys()))
    return rows


def run_forensics_benchmarks() -> dict:
    analyzer = ForensicsAnalyzer(prefer_gpu=True)
    image_real = analyzer.analyze_image(load_fixture("forensics_image_real.json"))
    image_fake = analyzer.analyze_image(load_fixture("forensics_image_fake.json"))
    video = analyzer.analyze_video(load_fixture("forensics_video.json"))
    audio = analyzer.analyze_audio(load_fixture("forensics_audio.json"))
    batch = analyzer.batch_analyze_images([load_fixture("forensics_image_fake.json") for _ in range(100)])
    rows = [
        {
            "asset_id": image_real["asset_id"],
            "manipulated": image_real["manipulated"],
            "detector": "image",
            "score": image_real["scores"]["high_frequency_ratio"],
            "device": analyzer.device,
        },
        {
            "asset_id": image_fake["asset_id"],
            "manipulated": image_fake["manipulated"],
            "detector": "image",
            "score": image_fake["scores"]["high_frequency_ratio"],
            "device": analyzer.device,
        },
        {"asset_id": video["asset_id"], "manipulated": video["manipulated"], "detector": "video", "score": video["temporal_inconsistency"], "device": analyzer.device},
        {"asset_id": audio["asset_id"], "manipulated": audio["manipulated"], "detector": "audio", "score": audio["splice_score"], "device": analyzer.device},
        {"asset_id": "batch_100", "manipulated": batch["manipulated"], "detector": "batch_images", "score": batch["throughput_per_sec"], "device": batch["device"]},
    ]
    out_root = results_dir() / "forensics"
    _write_csv(out_root / "image_detection_results.csv", rows, list(rows[0].keys()))
    (out_root / "causal_forensics_chain.json").write_text(json.dumps(image_fake["causal_chain"], indent=2))
    return {"image_real": image_real, "image_fake": image_fake, "video": video, "audio": audio, "batch": batch}


def run_oracle_benchmarks() -> dict:
    rows: list[dict] = []
    sample_explanations: list[dict] = []
    for seed in SEEDS:
        docs, queries = generate_synthetic_collection(seed, num_docs=150, num_queries=18)
        dense = DenseRetriever(docs, prefer_gpu=True)
        sparse = ClassicalIndex(docs)
        oracle = OracleRanker(docs)
        for query in queries:
            dense_ranked = dense.search(query["text"])
            oracle_ranked = oracle.rerank(dense_ranked, query["text"])
            bm25_ranked = sparse.rank(query["text"], "bm25")
            standard_top = dense_ranked[0][0]
            oracle_top = oracle_ranked[0][0]
            rows.append(
                {
                    "seed": seed,
                    "query_id": query["id"],
                    "bm25_top": bm25_ranked[0][0],
                    "dense_top": standard_top,
                    "oracle_top": oracle_top,
                    "oracle_score": round(oracle_ranked[0][1], 4),
                    "oracle_beats_dense": oracle_top != standard_top,
                    "oracle_beats_bm25": oracle_top != bm25_ranked[0][0],
                }
            )
            counterfactual = oracle.counterfactual(oracle_ranked, oracle_top)
            if len(sample_explanations) < 6:
                sample_explanations.append(
                    {
                        "seed": seed,
                        "query_id": query["id"],
                        "doc_id": oracle_top,
                        "explanation": oracle_ranked[0][2].summary,
                        "factors": oracle_ranked[0][2].factors,
                        "counterfactual": counterfactual,
                    }
                )
    out_root = results_dir() / "themis_oracle"
    _write_csv(out_root / "causal_vs_standard.csv", rows, list(rows[0].keys()))
    (out_root / "explanations_sample.json").write_text(json.dumps(sample_explanations, indent=2))
    return {"rows": rows, "explanations": sample_explanations}


def run_external_benchmarks(
    *,
    ir_manifest_path: str | None = None,
    passages_tsv: str | None = None,
    queries_tsv: str | None = None,
    qrels_tsv: str | None = None,
    forensics_manifest_path: str | None = None,
) -> dict:
    status = external_resource_status(
        {
            "external_datasets": {
                "ir_manifest": ir_manifest_path or "",
                "forensics_manifest": forensics_manifest_path or "",
                "msmarco": {
                    "passages_tsv": passages_tsv or "",
                    "queries_tsv": queries_tsv or "",
                    "qrels_tsv": qrels_tsv or "",
                },
            }
        }
    )
    summary: dict[str, object] = {"status": status, "ir": None, "forensics": None}

    ir_bundle = load_optional_ir_collection(
        manifest_path=ir_manifest_path,
        passages_tsv=passages_tsv,
        queries_tsv=queries_tsv,
        qrels_tsv=qrels_tsv,
        max_docs=1000,
        max_queries=100,
    )
    if ir_bundle is not None and ir_bundle["documents"] and ir_bundle["queries"]:
        docs = ir_bundle["documents"]
        queries = ir_bundle["queries"]
        index = ClassicalIndex(docs)
        dense = DenseRetriever(docs, prefer_gpu=True)
        cross = CrossEncoderReranker(docs, prefer_gpu=True)
        validation_rows = [
            (index.bm25_scores(query["text"]), dense.search(query["text"]), set(query["relevant"])) for query in queries[: min(len(queries), 8)]
        ]
        alpha = tune_alpha(validation_rows) if validation_rows else 0.55
        bm25_maps = []
        dense_ndcgs = []
        hybrid_ndcgs = []
        oracle_wins = []
        oracle = OracleRanker(docs)
        for query in queries:
            relevant = set(query["relevant"])
            bm25_ranked = index.rank(query["text"], "bm25")
            dense_ranked = dense.search(query["text"])
            reranked = cross.rerank(query["text"], dense_ranked)
            hybrid_ranked = hybrid_fusion(index.bm25_scores(query["text"]), reranked, alpha)
            oracle_ranked = oracle.rerank(dense_ranked, query["text"])
            bm25_maps.append(average_precision(_ranked_doc_ids(bm25_ranked), relevant))
            dense_ndcgs.append(ndcg_at_k(_ranked_doc_ids(dense_ranked), relevant, 3))
            hybrid_ndcgs.append(ndcg_at_k(_ranked_doc_ids(hybrid_ranked), relevant, 3))
            oracle_wins.append(1.0 if oracle_ranked[0][0] != dense_ranked[0][0] else 0.0)
        ir_summary = {
            "dataset": ir_bundle["dataset"],
            "documents": len(docs),
            "queries": len(queries),
            "bm25_mean_map": round(sum(bm25_maps) / max(len(bm25_maps), 1), 4),
            "dense_mean_ndcg@3": round(sum(dense_ndcgs) / max(len(dense_ndcgs), 1), 4),
            "hybrid_mean_ndcg@3": round(sum(hybrid_ndcgs) / max(len(hybrid_ndcgs), 1), 4),
            "oracle_changes_dense_top_pct": round((sum(oracle_wins) / max(len(oracle_wins), 1)) * 100, 2),
            "dense_backend": dense.index_stats()["backend"],
            "dense_device": dense.index_stats()["device"],
            "hybrid_alpha": alpha,
        }
        out_root = results_dir()
        (out_root / "ir" / "external_collection_summary.json").write_text(json.dumps(ir_summary, indent=2))
        (out_root / "neural_search" / "external_collection_summary.json").write_text(json.dumps(ir_summary, indent=2))
        (out_root / "themis_oracle" / "external_collection_summary.json").write_text(json.dumps(ir_summary, indent=2))
        summary["ir"] = ir_summary

    forensics_bundle = load_optional_forensics_collection(manifest_path=forensics_manifest_path)
    if forensics_bundle is not None:
        analyzer = ForensicsAnalyzer(prefer_gpu=True)
        image_results = []
        video_results = []
        audio_results = []
        for item in forensics_bundle["images"]:
            result = analyzer.analyze_image(item["payload"])
            image_results.append({"id": item["id"], "expected": item["label"], "predicted": result["manipulated"]})
        for item in forensics_bundle["videos"]:
            result = analyzer.analyze_video(item["payload"])
            video_results.append({"id": item["id"], "expected": item["label"], "predicted": result["manipulated"]})
        for item in forensics_bundle["audio"]:
            result = analyzer.analyze_audio(item["payload"])
            audio_results.append({"id": item["id"], "expected": item["label"], "predicted": result["manipulated"]})

        all_results = image_results + video_results + audio_results
        labeled = [item for item in all_results if item["expected"] is not None]
        accuracy = (
            sum(1.0 for item in labeled if bool(item["expected"]) == bool(item["predicted"])) / len(labeled)
            if labeled
            else None
        )
        forensic_summary = {
            "dataset": forensics_bundle["dataset"],
            "device": analyzer.device,
            "images": len(image_results),
            "videos": len(video_results),
            "audio": len(audio_results),
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
        }
        (results_dir() / "forensics" / "external_dataset_summary.json").write_text(json.dumps(forensic_summary, indent=2))
        summary["forensics"] = forensic_summary

    return summary


def run_all_benchmarks() -> dict:
    return {
        "ir": run_ir_benchmarks(),
        "neural": run_neural_benchmarks(),
        "forensics": run_forensics_benchmarks(),
        "oracle": run_oracle_benchmarks(),
        "external": run_external_benchmarks(),
    }
