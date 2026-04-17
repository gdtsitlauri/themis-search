from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CausalEdge:
    source: str
    target: str
    strength: float


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def partial_correlation(x: np.ndarray, y: np.ndarray, cond: np.ndarray | None = None) -> float:
    if cond is None or cond.size == 0:
        return _safe_corr(x, y)
    cond_2d = np.atleast_2d(cond)
    if cond_2d.shape[0] != x.shape[0]:
        cond_2d = cond_2d.T
    design = np.column_stack([np.ones(x.shape[0]), cond_2d])
    beta_x, *_ = np.linalg.lstsq(design, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid_x = x - design @ beta_x
    resid_y = y - design @ beta_y
    return _safe_corr(resid_x, resid_y)


def pc_skeleton(
    feature_names: list[str],
    samples: np.ndarray,
    target_name: str | None = None,
    corr_threshold: float = 0.12,
    max_conditioning: int = 1,
) -> list[CausalEdge]:
    if samples.ndim != 2 or samples.shape[0] < 3 or samples.shape[1] != len(feature_names):
        return []
    keep: list[CausalEdge] = []
    for i, j in itertools.combinations(range(len(feature_names)), 2):
        x = samples[:, i]
        y = samples[:, j]
        base = abs(_safe_corr(x, y))
        if base < corr_threshold:
            continue
        independent = False
        cond_candidates = [k for k in range(len(feature_names)) if k not in {i, j}]
        for order in range(1, min(max_conditioning, len(cond_candidates)) + 1):
            for subset in itertools.combinations(cond_candidates, order):
                cond = samples[:, subset]
                if abs(partial_correlation(x, y, cond)) < corr_threshold:
                    independent = True
                    break
            if independent:
                break
        if not independent:
            left = feature_names[i]
            right = feature_names[j]
            if target_name is not None:
                if right == target_name and left != target_name:
                    keep.append(CausalEdge(left, right, round(base, 4)))
                elif left == target_name and right != target_name:
                    keep.append(CausalEdge(right, left, round(base, 4)))
                else:
                    keep.append(CausalEdge(left, right, round(base, 4)))
            else:
                keep.append(CausalEdge(left, right, round(base, 4)))
    return keep


def feature_weights_from_edges(
    feature_names: list[str],
    samples: np.ndarray,
    target_name: str,
    edges: list[CausalEdge],
) -> dict[str, float]:
    if target_name not in feature_names or samples.ndim != 2:
        return {}
    target_idx = feature_names.index(target_name)
    target = samples[:, target_idx]
    weights: dict[str, float] = {}
    for edge in edges:
        if edge.target != target_name:
            continue
        source_idx = feature_names.index(edge.source)
        strength = abs(_safe_corr(samples[:, source_idx], target))
        weights[edge.source] = strength + abs(edge.strength)
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in weights.items()}
