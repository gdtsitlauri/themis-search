from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from themis.query_language import normalize_external_ast, parse_query_text, payload_from_ast
from themis.schemas import QueryPayload


def _fallback_parse(query: str) -> QueryPayload:
    try:
        ast = parse_query_text(query)
        return QueryPayload(**payload_from_ast(query, ast))
    except Exception as exc:
        return QueryPayload(
            raw=query,
            normalized=query,
            ast={"type": "RawQuery", "tokens": query.split()},
            errors=[str(exc)],
        )


@lru_cache(maxsize=1)
def _query_executable() -> str | None:
    env_path = os.getenv("THEMIS_QUERY_BIN")
    if env_path and Path(env_path).exists():
        return env_path
    installed = shutil.which("themis-query")
    if installed:
        return installed
    repo_root = Path(__file__).resolve().parents[3]
    for candidate in (repo_root / "src" / "haskell" / "dist-newstyle").rglob("themis-query"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    cabal = shutil.which("cabal")
    if not cabal:
        return None
    try:
        completed = subprocess.run(
            [cabal, "list-bin", "themis-query"],
            cwd=repo_root / "src" / "haskell",
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        path = completed.stdout.strip()
        if path and Path(path).exists():
            return path
    except Exception:
        return None
    return None


def parse_query(query: str) -> QueryPayload:
    executable = _query_executable()
    repo_root = Path(__file__).resolve().parents[3]
    if not executable:
        return _fallback_parse(query)
    try:
        completed = subprocess.run(
            [executable, query],
            cwd=repo_root / "src" / "haskell",
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        payload = json.loads(completed.stdout)
        payload["ast"] = normalize_external_ast(payload.get("ast", {}))
        payload["normalized"] = payload.get("normalized") or query
        return QueryPayload(**payload)
    except Exception as exc:  # pragma: no cover - depends on toolchain
        fallback = _fallback_parse(query)
        fallback.errors.append(str(exc))
        return fallback
