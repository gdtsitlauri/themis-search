from __future__ import annotations

import os
import math
from collections import Counter

import numpy as np

from themis.ir import tokenize

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None
    CrossEncoder = None

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


_SENTENCE_MODEL_CACHE: dict[tuple[str, str, bool], object] = {}
_CROSS_MODEL_CACHE: dict[tuple[str, str, bool], object] = {}


def _hash_embed(text: str, dims: int = 64) -> np.ndarray:
    vec = np.zeros(dims, dtype=np.float32)
    for token in tokenize(text):
        vec[hash(token) % dims] += 1.0
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


class DenseRetriever:
    def __init__(
        self,
        documents: list[dict],
        model_name: str = "all-MiniLM-L6-v2",
        prefer_gpu: bool = False,
        allow_remote_models: bool | None = None,
        batch_size: int = 16,
    ):
        self.documents = {doc["id"]: doc for doc in documents}
        self.doc_ids = list(self.documents)
        self.model_name = model_name
        self.batch_size = batch_size
        self.allow_remote_models = bool(
            os.getenv("THEMIS_ALLOW_REMOTE_MODELS") == "1" if allow_remote_models is None else allow_remote_models
        )
        self.device = "cuda" if prefer_gpu and torch is not None and torch.cuda.is_available() else "cpu"
        self.model = None
        self.backend = "hash"
        if SentenceTransformer is not None:
            try:
                cache_key = (model_name, self.device, self.allow_remote_models)
                if cache_key not in _SENTENCE_MODEL_CACHE:
                    _SENTENCE_MODEL_CACHE[cache_key] = SentenceTransformer(
                        model_name,
                        local_files_only=not self.allow_remote_models,
                        device=self.device,
                    )
                self.model = _SENTENCE_MODEL_CACHE[cache_key]
                self.backend = "sentence_transformer"
            except Exception:
                self.model = None
        texts = [f"{doc['title']} {doc['text']}" for doc in documents]
        matrix = self.encode_texts(texts)
        self.doc_embeddings = {doc_id: matrix[idx] for idx, doc_id in enumerate(self.doc_ids)}
        self.doc_matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.doc_tensor = None
        if self.device == "cuda" and torch is not None and self.doc_matrix.size:
            self.doc_tensor = torch.tensor(self.doc_matrix, device=self.device, dtype=torch.float32)
        self.exact_index = None
        self.approx_index = None
        self._build_indexes()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.ascontiguousarray(embeddings, dtype=np.float32)
        return np.ascontiguousarray([_hash_embed(text, dims=256) for text in texts], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def _build_indexes(self) -> None:
        if faiss is None or self.doc_matrix.size == 0:
            return
        dim = self.doc_matrix.shape[1]
        flat_index = faiss.IndexFlatIP(dim)
        flat_index.add(self.doc_matrix)
        self.exact_index = flat_index
        if len(self.doc_ids) >= 80:
            nlist = max(2, min(int(math.sqrt(len(self.doc_ids))), max(2, len(self.doc_ids) // 40)))
            quantizer = faiss.IndexFlatIP(dim)
            approx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            approx.train(self.doc_matrix)
            approx.add(self.doc_matrix)
            approx.nprobe = min(4, nlist)
            self.approx_index = approx

    def search(self, query: str, top_k: int = 5, approximate: bool = False) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        q = np.ascontiguousarray(self.encode_texts([query]), dtype=np.float32)
        if self.device == "cuda" and not approximate and self.doc_tensor is not None and torch is not None and not hasattr(faiss, "StandardGpuResources"):
            query_tensor = torch.tensor(q, device=self.device, dtype=torch.float32)
            scores = torch.matmul(query_tensor, self.doc_tensor.T)[0]
            values, indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
            return [(self.doc_ids[int(idx)], float(score)) for score, idx in zip(values.tolist(), indices.tolist(), strict=True)]
        index = self.approx_index if approximate and self.approx_index is not None else self.exact_index
        if index is not None:
            scores, ids = index.search(q, min(top_k, len(self.doc_ids)))
            results = []
            for score, idx in zip(scores[0], ids[0], strict=True):
                if idx < 0:
                    continue
                results.append((self.doc_ids[int(idx)], float(score)))
            return results
        scored = []
        for doc_id, emb in self.doc_embeddings.items():
            scored.append((doc_id, float(np.dot(q[0], emb))))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]

    def index_stats(self) -> dict[str, object]:
        return {
            "documents": len(self.doc_ids),
            "embedding_dim": int(self.doc_matrix.shape[1]) if self.doc_matrix.size else 0,
            "backend": self.backend,
            "device": self.device,
            "faiss": faiss is not None,
            "faiss_gpu": bool(faiss is not None and hasattr(faiss, "StandardGpuResources")),
            "gpu_exact_fallback": bool(self.device == "cuda" and self.doc_tensor is not None),
            "approximate_index": self.approx_index is not None,
        }


class CrossEncoderReranker:
    def __init__(
        self,
        documents: list[dict],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
        allow_remote_models: bool | None = None,
        prefer_gpu: bool = False,
    ):
        self.documents = {doc["id"]: doc for doc in documents}
        self.allow_remote_models = bool(
            os.getenv("THEMIS_ALLOW_REMOTE_MODELS") == "1" if allow_remote_models is None else allow_remote_models
        )
        self.device = "cuda" if prefer_gpu and torch is not None and torch.cuda.is_available() else "cpu"
        self.model = None
        self.backend = "heuristic"
        if CrossEncoder is not None:
            try:
                cache_key = (model_name, self.device, self.allow_remote_models)
                if cache_key not in _CROSS_MODEL_CACHE:
                    _CROSS_MODEL_CACHE[cache_key] = CrossEncoder(
                        model_name,
                        local_files_only=not self.allow_remote_models,
                        device=self.device,
                    )
                self.model = _CROSS_MODEL_CACHE[cache_key]
                self.backend = "cross_encoder"
            except Exception:
                self.model = None

    def rerank(self, query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if self.model is not None and candidates:
            pairs = [[query, f"{self.documents[doc_id]['title']} {self.documents[doc_id]['text']}"] for doc_id, _ in candidates]
            try:
                model_scores = np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=np.float32)
                rescored = []
                for (doc_id, base_score), model_score in zip(candidates, model_scores.tolist(), strict=True):
                    rescored.append((doc_id, float(0.65 * model_score + 0.35 * base_score)))
                return sorted(rescored, key=lambda item: (-item[1], item[0]))
            except Exception:
                pass
        query_terms = Counter(tokenize(query))
        rescored = []
        for doc_id, base_score in candidates:
            title_tokens = Counter(tokenize(self.documents[doc_id]["title"]))
            body_tokens = Counter(tokenize(self.documents[doc_id]["text"]))
            overlap = sum((query_terms & body_tokens).values())
            title_overlap = sum((query_terms & title_tokens).values())
            phrase_bonus = 0.08 if query.lower() in self.documents[doc_id]["text"].lower() else 0.0
            rescored.append((doc_id, float(base_score + 0.07 * overlap + 0.12 * title_overlap + phrase_bonus)))
        return sorted(rescored, key=lambda item: (-item[1], item[0]))


class LateInteractionRetriever:
    def __init__(self, documents: list[dict], prefer_gpu: bool = False):
        self.documents = {doc["id"]: doc for doc in documents}
        self.device = "cuda" if prefer_gpu and torch is not None and torch.cuda.is_available() else "cpu"
        self.token_vectors = {
            doc["id"]: np.ascontiguousarray([_hash_embed(token, dims=64) for token in tokenize(doc["text"])], dtype=np.float32)
            for doc in documents
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_tokens = np.ascontiguousarray([_hash_embed(token, dims=64) for token in tokenize(query)], dtype=np.float32)
        scored = []
        for doc_id, doc_tokens in self.token_vectors.items():
            if doc_tokens.size == 0 or query_tokens.size == 0:
                score = 0.0
            elif torch is not None:
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                q_tensor = torch.tensor(query_tokens, device=self.device, dtype=dtype)
                d_tensor = torch.tensor(doc_tokens, device=self.device, dtype=dtype)
                score = float((q_tensor @ d_tensor.T).max(dim=1).values.mean().item())
            else:
                sims = query_tokens @ doc_tokens.T
                score = float(np.max(sims, axis=1).mean())
            scored.append((doc_id, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def hybrid_fusion(
    sparse_scores: dict[str, float],
    dense_scores: list[tuple[str, float]],
    alpha: float = 0.55,
) -> list[tuple[str, float]]:
    dense_map = dict(dense_scores)
    all_docs = set(sparse_scores) | set(dense_map)
    fused = []
    for doc_id in all_docs:
        fused_score = alpha * sparse_scores.get(doc_id, 0.0) + (1 - alpha) * dense_map.get(doc_id, 0.0)
        fused.append((doc_id, fused_score))
    return sorted(fused, key=lambda item: (-item[1], item[0]))


def tune_alpha(validation_rows: list[tuple[dict[str, float], list[tuple[str, float]], set[str]]]) -> float:
    best_alpha = 0.5
    best_score = -math.inf
    for alpha in (0.2, 0.35, 0.5, 0.55, 0.65, 0.8):
        score = 0.0
        for sparse, dense, relevant in validation_rows:
            fused = [doc_id for doc_id, _ in hybrid_fusion(sparse, dense, alpha)[:3]]
            score += sum(1.0 for doc_id in fused if doc_id in relevant)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def warm_model_cache(prefer_gpu: bool = True) -> dict[str, object]:
    dense = DenseRetriever([], prefer_gpu=prefer_gpu)
    cross = CrossEncoderReranker([], prefer_gpu=prefer_gpu)
    return {
        "dense_backend": dense.backend,
        "dense_device": dense.device,
        "cross_backend": cross.backend,
        "cross_device": cross.device,
    }
