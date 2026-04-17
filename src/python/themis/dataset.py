from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "data" / "fixtures"


def _load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def load_documents() -> list[dict[str, Any]]:
    return _load_json("documents.json")


def load_queries() -> list[dict[str, Any]]:
    return _load_json("queries.json")


def load_fixture(name: str) -> dict[str, Any]:
    return _load_json(name)


def results_dir() -> Path:
    path = ROOT / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return ROOT / "configs" / "app.json"


def load_app_config() -> dict[str, Any]:
    path = config_path()
    return json.loads(path.read_text()) if path.exists() else {}


def resolve_local_path(path_like: str | None, base: Path | None = None) -> Path | None:
    if not path_like:
        return None
    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = (base / candidate) if base is not None else (ROOT / candidate)
    candidate = candidate.resolve()
    return candidate if candidate.exists() else None


def external_resource_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_app_config()
    external = config.get("external_datasets", {})
    msmarco = external.get("msmarco", {})
    entries = {
        "ir_manifest": resolve_local_path(external.get("ir_manifest")),
        "forensics_manifest": resolve_local_path(external.get("forensics_manifest")),
        "msmarco_passages": resolve_local_path(msmarco.get("passages_tsv")),
        "msmarco_queries": resolve_local_path(msmarco.get("queries_tsv")),
        "msmarco_qrels": resolve_local_path(msmarco.get("qrels_tsv")),
    }
    return {
        key: {
            "available": path is not None,
            "path": str(path) if path is not None else None,
        }
        for key, path in entries.items()
    }


def load_ir_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    documents = [
        {
            "id": doc["id"],
            "title": doc.get("title", doc["id"]),
            "text": doc["text"],
            "metadata": doc.get("metadata", {}),
        }
        for doc in payload.get("documents", [])
    ]
    queries = [
        {
            "id": query["id"],
            "text": query["text"],
            "relevant": list(query.get("relevant", [])),
        }
        for query in payload.get("queries", [])
    ]
    return {"dataset": payload.get("dataset", manifest_path.stem), "documents": documents, "queries": queries}


def load_msmarco_subset(
    passages_tsv: str | Path,
    queries_tsv: str | Path,
    qrels_tsv: str | Path,
    max_docs: int = 1000,
    max_queries: int = 100,
) -> dict[str, Any]:
    qrels: dict[str, list[str]] = {}
    with Path(qrels_tsv).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) >= 4:
                qid, _, pid, rel = row[:4]
                if rel and float(rel) <= 0:
                    continue
            elif len(row) >= 2:
                qid, pid = row[:2]
            else:
                continue
            qrels.setdefault(qid, []).append(pid)
    selected_qids = list(qrels)[:max_queries]
    selected_pids: list[str] = []
    for qid in selected_qids:
        for pid in qrels.get(qid, []):
            if pid not in selected_pids:
                selected_pids.append(pid)
            if len(selected_pids) >= max_docs:
                break
        if len(selected_pids) >= max_docs:
            break
    selected_pid_set = set(selected_pids)

    documents = []
    with Path(passages_tsv).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            pid = row[0]
            if selected_pid_set and pid not in selected_pid_set:
                continue
            if len(row) == 2:
                title = f"passage {pid}"
                text = row[1]
            else:
                title = row[1] or f"passage {pid}"
                text = row[-1]
            documents.append({"id": pid, "title": title, "text": text, "metadata": {"source": "msmarco_local"}})
            if len(documents) >= max_docs:
                break

    doc_id_set = {doc["id"] for doc in documents}
    queries = []
    with Path(queries_tsv).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            qid, text = row[0], row[1]
            if qid not in selected_qids:
                continue
            queries.append(
                {
                    "id": qid,
                    "text": text,
                    "relevant": [pid for pid in qrels.get(qid, []) if pid in doc_id_set],
                }
            )
    return {"dataset": "msmarco_local_subset", "documents": documents, "queries": queries}


def load_optional_ir_collection(
    *,
    config: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    passages_tsv: str | Path | None = None,
    queries_tsv: str | Path | None = None,
    qrels_tsv: str | Path | None = None,
    max_docs: int = 1000,
    max_queries: int = 100,
) -> dict[str, Any] | None:
    config = config or load_app_config()
    external = config.get("external_datasets", {})
    msmarco = external.get("msmarco", {})
    manifest = resolve_local_path(str(manifest_path) if manifest_path is not None else external.get("ir_manifest"))
    if manifest is not None:
        return load_ir_manifest(manifest)
    passages = resolve_local_path(str(passages_tsv) if passages_tsv is not None else msmarco.get("passages_tsv"))
    queries = resolve_local_path(str(queries_tsv) if queries_tsv is not None else msmarco.get("queries_tsv"))
    qrels = resolve_local_path(str(qrels_tsv) if qrels_tsv is not None else msmarco.get("qrels_tsv"))
    if passages is not None and queries is not None and qrels is not None:
        return load_msmarco_subset(passages, queries, qrels, max_docs=max_docs, max_queries=max_queries)
    return None


def load_forensics_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    base = manifest_path.parent

    def _load_entries(kind: str) -> list[dict[str, Any]]:
        loaded = []
        for entry in payload.get(kind, []):
            local_path = resolve_local_path(entry["path"], base)
            if local_path is None:
                continue
            loaded.append(
                {
                    "id": entry.get("id", local_path.stem),
                    "path": str(local_path),
                    "label": entry.get("label"),
                    "payload": json.loads(local_path.read_text()),
                }
            )
        return loaded

    return {
        "dataset": payload.get("dataset", manifest_path.stem),
        "images": _load_entries("images"),
        "videos": _load_entries("videos"),
        "audio": _load_entries("audio"),
    }


def load_optional_forensics_collection(
    *,
    config: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any] | None:
    config = config or load_app_config()
    external = config.get("external_datasets", {})
    manifest = resolve_local_path(str(manifest_path) if manifest_path is not None else external.get("forensics_manifest"))
    if manifest is None:
        return None
    return load_forensics_manifest(manifest)


TOPIC_BANK = [
    {
        "slug": "causal",
        "title": "causal search",
        "terms": ["causal relevance", "citation influence", "counterfactual retrieval", "upstream evidence"],
        "queries": ["causal relevance search", "citation influence retrieval", "counterfactual document ranking"],
    },
    {
        "slug": "hybrid",
        "title": "hybrid search",
        "terms": ["dense retrieval", "bm25 fusion", "semantic ranking", "faiss indexing"],
        "queries": ["hybrid semantic retrieval", "dense bm25 fusion", "faiss neural search"],
    },
    {
        "slug": "image",
        "title": "image forensics",
        "terms": ["copy move detection", "splicing artifacts", "spectrum analysis", "gan fingerprint"],
        "queries": ["image forgery detection", "copy move spectrum artifact", "synthetic image fingerprint"],
    },
    {
        "slug": "video",
        "title": "video forensics",
        "terms": ["frame duplication", "temporal inconsistency", "compression artifacts", "dct traces"],
        "queries": ["video frame duplication", "temporal manipulation detection", "compression artifact analysis"],
    },
    {
        "slug": "audio",
        "title": "audio forensics",
        "terms": ["audio splicing", "background inconsistency", "voice conversion", "frequency analysis"],
        "queries": ["audio splicing detection", "background noise inconsistency", "voice conversion artifacts"],
    },
]


def generate_synthetic_collection(
    seed: int,
    num_docs: int = 250,
    num_queries: int = 45,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    documents: list[dict[str, Any]] = []
    topic_docs: dict[str, list[str]] = {topic["slug"]: [] for topic in TOPIC_BANK}
    role_templates = [
        "foundational study",
        "method paper",
        "evaluation note",
        "survey summary",
        "application report",
    ]
    for idx in range(num_docs):
        topic = TOPIC_BANK[idx % len(TOPIC_BANK)]
        role = role_templates[idx % len(role_templates)]
        term_a, term_b = rng.sample(topic["terms"], k=2)
        doc_id = f"s{seed}_d{idx:04d}"
        prior_docs = topic_docs[topic["slug"]][:]
        citations = prior_docs[-2:]
        year = 2010 + (idx % 13)
        title = f"{topic['title'].title()} {role.title()} {idx}"
        text = (
            f"This {role} discusses {term_a}, {term_b}, and {topic['slug']} signals. "
            f"It compares lexical retrieval with semantic ranking and records {topic['slug']} evidence. "
            f"Earlier sources influence later findings through causal and temporal links."
        )
        documents.append(
            {
                "id": doc_id,
                "title": title,
                "text": text,
                "metadata": {
                    "year": year,
                    "citations": citations,
                    "pagerank": round(0.1 + ((idx % 17) / 50), 3),
                    "tags": [topic["slug"], role.split()[0], "synthetic"],
                },
            }
        )
        topic_docs[topic["slug"]].append(doc_id)

    queries: list[dict[str, Any]] = []
    for idx in range(num_queries):
        topic = TOPIC_BANK[idx % len(TOPIC_BANK)]
        query_text = topic["queries"][idx % len(topic["queries"])]
        relevant_pool = topic_docs[topic["slug"]][: max(5, len(topic_docs[topic["slug"]]) // 3)]
        queries.append(
            {
                "id": f"s{seed}_q{idx:03d}",
                "text": query_text,
                "relevant": relevant_pool[:5],
            }
        )
    return documents, queries


def expand_documents(documents: list[dict[str, Any]], target_size: int) -> list[dict[str, Any]]:
    if len(documents) >= target_size:
        return documents[:target_size]
    expanded: list[dict[str, Any]] = []
    multiplier = 0
    while len(expanded) < target_size:
        for doc in documents:
            clone = json.loads(json.dumps(doc))
            clone["id"] = f"{doc['id']}_x{multiplier}"
            clone["metadata"]["citations"] = [f"{item}_x{max(multiplier - 1, 0)}" for item in clone["metadata"].get("citations", [])]
            expanded.append(clone)
            if len(expanded) >= target_size:
                break
        multiplier += 1
    return expanded
