from __future__ import annotations

import argparse
import json
from pathlib import Path

from themis.api import get_index_stats, search
from themis.benchmarks import run_all_benchmarks, run_external_benchmarks, run_forensics_benchmarks, run_oracle_benchmarks
from themis.dataset import external_resource_status, load_fixture
from themis.forensics import ForensicsAnalyzer
from themis.neural import warm_model_cache


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="themis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)

    subparsers.add_parser("benchmark")
    subparsers.add_parser("dataset-status")
    subparsers.add_parser("oracle-eval")
    subparsers.add_parser("warm-models")

    external_parser = subparsers.add_parser("benchmark-external")
    external_parser.add_argument("--ir-manifest", type=str, default=None)
    external_parser.add_argument("--msmarco-passages", type=str, default=None)
    external_parser.add_argument("--msmarco-queries", type=str, default=None)
    external_parser.add_argument("--msmarco-qrels", type=str, default=None)
    external_parser.add_argument("--forensics-manifest", type=str, default=None)

    forensics_parser = subparsers.add_parser("forensics")
    forensics_parser.add_argument("--image", type=str, default="data/fixtures/forensics_image_fake.json")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "index":
        print(json.dumps(get_index_stats(), indent=2))
        return
    if args.command == "search":
        from themis.api import SearchRequest

        payload = search(SearchRequest(query=args.query, top_k=args.top_k))
        print(json.dumps(payload, indent=2))
        return
    if args.command == "benchmark":
        print(json.dumps(run_all_benchmarks(), indent=2))
        return
    if args.command == "dataset-status":
        print(json.dumps(external_resource_status(), indent=2))
        return
    if args.command == "oracle-eval":
        print(json.dumps(run_oracle_benchmarks(), indent=2))
        return
    if args.command == "benchmark-external":
        print(
            json.dumps(
                run_external_benchmarks(
                    ir_manifest_path=args.ir_manifest,
                    passages_tsv=args.msmarco_passages,
                    queries_tsv=args.msmarco_queries,
                    qrels_tsv=args.msmarco_qrels,
                    forensics_manifest_path=args.forensics_manifest,
                ),
                indent=2,
            )
        )
        return
    if args.command == "warm-models":
        print(json.dumps(warm_model_cache(), indent=2))
        return
    if args.command == "forensics":
        analyzer = ForensicsAnalyzer()
        payload = load_fixture(Path(args.image).name)
        print(json.dumps(analyzer.analyze_image(payload), indent=2))
        return


if __name__ == "__main__":
    main()
