from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

from themis.api import SearchRequest, search
from themis.causal import pc_skeleton
from themis.dataset import external_resource_status, load_documents, load_fixture, load_forensics_manifest, load_ir_manifest
from themis.forensics import ForensicsAnalyzer
from themis.ir import ClassicalIndex
from themis.neural import DenseRetriever, hybrid_fusion
from themis.oracle import OracleRanker
from themis.query_language import estimate_selectivity, parse_query_text


class ThemisTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents()
        cls.index = ClassicalIndex(cls.documents)

    def test_inverted_index(self) -> None:
        postings = dict(self.index.term_lookup("causal"))
        self.assertIn("d1", postings)
        self.assertGreaterEqual(postings["d1"], 1)

    def test_bm25_ranking(self) -> None:
        ranked = self.index.rank("causal relevance search", "bm25")
        top3 = [doc_id for doc_id, _ in ranked[:3]]
        self.assertIn("d1", top3)

    def test_query_language_and_near(self) -> None:
        ast = parse_query_text('"neural network" NEAR/5 training')
        self.assertEqual(ast["type"], "Near")
        self.assertEqual(ast["distance"], 5)
        implicit = parse_query_text('causal relevance AND "counterfactual ranking"')
        self.assertEqual(implicit["type"], "And")
        self.assertGreater(estimate_selectivity(implicit), 0.0)
        self.assertLess(estimate_selectivity(implicit), 1.0)

    def test_external_manifest_loaders(self) -> None:
        root = Path(__file__).resolve().parents[1]
        ir_bundle = load_ir_manifest(root / "data" / "fixtures" / "external_ir_manifest.json")
        self.assertEqual(ir_bundle["dataset"], "fixture_local_ir_manifest")
        self.assertEqual(len(ir_bundle["documents"]), 2)
        self.assertEqual(ir_bundle["queries"][0]["relevant"], ["ext_d1"])

        forensics_bundle = load_forensics_manifest(root / "data" / "fixtures" / "external_forensics_manifest.json")
        self.assertEqual(forensics_bundle["dataset"], "fixture_local_forensics_manifest")
        self.assertEqual(len(forensics_bundle["images"]), 2)
        self.assertEqual(len(forensics_bundle["videos"]), 1)
        self.assertEqual(len(forensics_bundle["audio"]), 1)

        status = external_resource_status(
            {
                "external_datasets": {
                    "ir_manifest": "data/fixtures/external_ir_manifest.json",
                    "forensics_manifest": "data/fixtures/external_forensics_manifest.json",
                    "msmarco": {"passages_tsv": "", "queries_tsv": "", "qrels_tsv": ""},
                }
            }
        )
        self.assertTrue(status["ir_manifest"]["available"])
        self.assertTrue(status["forensics_manifest"]["available"])

    def test_dense_retrieval(self) -> None:
        dense = DenseRetriever(self.documents)
        ranked = [doc_id for doc_id, _ in dense.search("semantic paraphrase retrieval", top_k=3)]
        self.assertIn("d2", ranked)

    def test_hybrid_search(self) -> None:
        dense = DenseRetriever(self.documents)
        sparse = self.index.bm25_scores("hybrid semantic retrieval")
        dense_ranked = dense.search("hybrid semantic retrieval")
        hybrid = [doc_id for doc_id, _ in hybrid_fusion(sparse, dense_ranked, alpha=0.55)[:3]]
        self.assertIn("d2", hybrid)

    def test_forensics_detection(self) -> None:
        analyzer = ForensicsAnalyzer()
        real = analyzer.analyze_image(load_fixture("forensics_image_real.json"))
        result = analyzer.analyze_image(load_fixture("forensics_image_fake.json"))
        self.assertFalse(real["manipulated"])
        self.assertTrue(result["manipulated"])
        self.assertTrue(result["causal_chain"]["artifacts"])
        self.assertIn("causal_parents", result["causal_chain"])

    def test_forensics_batch_analysis(self) -> None:
        analyzer = ForensicsAnalyzer()
        batch = analyzer.batch_analyze_images([load_fixture("forensics_image_fake.json") for _ in range(4)])
        self.assertEqual(batch["count"], 4)
        self.assertGreater(batch["throughput_per_sec"], 0)

    def test_causal_oracle(self) -> None:
        dense = DenseRetriever(self.documents)
        oracle = OracleRanker(self.documents)
        reranked = oracle.rerank(dense.search("causal relevance search"), "causal relevance search")
        scores = {doc_id: score for doc_id, score, _ in reranked}
        self.assertGreater(scores["d1"], scores["d2"])
        self.assertGreater(scores["d1"], scores.get("d5", 0.0))
        self.assertTrue(any("learned_weights=" in factor for factor in reranked[0][2].factors))

    def test_pc_skeleton_discovers_target_edges(self) -> None:
        samples = __import__("numpy").array(
            [
                [0.9, 0.8, 0.1, 1.7],
                [0.7, 0.6, 0.2, 1.3],
                [0.2, 0.1, 0.8, 0.5],
                [0.1, 0.2, 0.7, 0.4],
            ],
            dtype=float,
        )
        edges = pc_skeleton(["semantic", "citation", "noise", "target"], samples, target_name="target", corr_threshold=0.1)
        self.assertTrue(any(edge.target == "target" for edge in edges))

    def test_search_api(self) -> None:
        start = time.perf_counter()
        payload = search(SearchRequest(query='causal relevance AND "counterfactual ranking"', top_k=3))
        latency_ms = (time.perf_counter() - start) * 1000
        self.assertEqual(len(payload["results"]), 3)
        self.assertIn("timing_ms", payload)
        self.assertLess(latency_ms, 100.0)


if __name__ == "__main__":
    unittest.main()
