from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from themis.causal import feature_weights_from_edges, pc_skeleton

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


class ForensicsAnalyzer:
    def __init__(self, prefer_gpu: bool = False):
        self.device = "cuda" if prefer_gpu and torch is not None and torch.cuda.is_available() else "cpu"

    def _tensor(self, values: np.ndarray) -> torch.Tensor | None:
        if torch is None:
            return None
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        return torch.tensor(values, device=self.device, dtype=dtype)

    def _residual_map(self, pixels: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            blurred = cv2.GaussianBlur(pixels.astype(np.float32), (3, 3), 0)
            return pixels.astype(np.float32) - blurred
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        padded = np.pad(pixels, 1, mode="edge")
        residual = np.zeros_like(pixels, dtype=np.float32)
        for row in range(pixels.shape[0]):
            for col in range(pixels.shape[1]):
                residual[row, col] = float(np.sum(padded[row : row + 3, col : col + 3] * kernel))
        return residual

    def _copy_move_score(self, pixels: np.ndarray) -> float:
        block_size = 2 if min(pixels.shape) < 16 else 4
        seen: dict[tuple[float, ...], tuple[int, int]] = {}
        duplicates = 0
        for row in range(0, pixels.shape[0] - block_size + 1):
            for col in range(0, pixels.shape[1] - block_size + 1):
                block = tuple(np.round(pixels[row : row + block_size, col : col + block_size].flatten(), 2).tolist())
                if block in seen and np.mean(block) > 20:
                    duplicates += 1
                else:
                    seen[block] = (row, col)
        return float(min(1.0, duplicates / 3.0))

    def analyze_image(self, image_payload: dict) -> dict:
        pixels = np.array(image_payload["pixels"], dtype=np.float32)
        if torch is not None:
            tensor = self._tensor(pixels)
            diffs = float((torch.abs(tensor[1:] - tensor[:-1]).mean() + torch.abs(tensor[:, 1:] - tensor[:, :-1]).mean()).item())
        else:
            diffs = np.abs(np.diff(pixels, axis=0)).mean() + np.abs(np.diff(pixels, axis=1)).mean()
        residual = self._residual_map(pixels)
        copy_move_score = self._copy_move_score(pixels)
        if torch is not None:
            residual_tensor = self._tensor(residual)
            spectrum_tensor = torch.fft.fft2(residual_tensor.float())
            spectrum = torch.abs(spectrum_tensor).float()
            high_freq_ratio = float((spectrum[1:, 1:].mean() / (torch.abs(residual_tensor).mean() + 1e-6)).item())
            srm_score = float(torch.mean(torch.abs(residual_tensor)).item())
            peak_jump = float(torch.abs(torch.diff(self._tensor(pixels.flatten()))).max().item()) if pixels.size else 0.0
        else:
            spectrum = np.abs(np.fft.fft2(residual))
            high_freq_ratio = float(spectrum[1:, 1:].mean() / (np.abs(residual).mean() + 1e-6))
            srm_score = float(np.mean(np.abs(residual)))
            peak_jump = float(np.abs(np.diff(pixels.flatten())).max()) if pixels.size else 0.0
        manipulated = bool(
            copy_move_score > 0.2
            or max(diffs, peak_jump) > 20
            or srm_score > 4.0
            or (high_freq_ratio > 6.0 and srm_score > 2.0)
        )
        patch_samples = []
        for row in range(max(1, pixels.shape[0] - 1)):
            for col in range(max(1, pixels.shape[1] - 1)):
                patch = pixels[row : row + 2, col : col + 2]
                if patch.shape != (2, 2):
                    continue
                patch_residual = residual[row : row + 2, col : col + 2]
                patch_samples.append(
                    [
                        float(np.mean(np.abs(patch_residual))),
                        float(np.std(patch)),
                        float(np.max(patch) - np.min(patch)),
                        float(np.mean(patch)),
                        float(np.mean(np.abs(np.fft.fft2(patch_residual)))),
                        1.0 if np.mean(patch) > np.mean(pixels) else 0.0,
                    ]
                )
        feature_names = ["residual_energy", "local_std", "intensity_range", "mean_intensity", "frequency_energy", "manipulation_proxy"]
        patch_matrix = np.asarray(patch_samples, dtype=np.float32) if patch_samples else np.zeros((0, len(feature_names)), dtype=np.float32)
        causal_edges = pc_skeleton(feature_names, patch_matrix, target_name="manipulation_proxy", corr_threshold=0.08, max_conditioning=1)
        causal_weights = feature_weights_from_edges(feature_names, patch_matrix, "manipulation_proxy", causal_edges)
        reasons = []
        if copy_move_score > 0.2:
            reasons.append("duplicated_regions")
        if high_freq_ratio > 6.0 and srm_score > 2.0:
            reasons.append("frequency_artifacts")
        if srm_score > 4.0:
            reasons.append("srm_residual_anomaly")
        if diffs > 20 or peak_jump > 20:
            reasons.append("noise_inconsistency")
        if not reasons:
            reasons.append("no_strong_artifacts")
        return {
            "asset_id": image_payload["id"],
            "manipulated": manipulated,
            "scores": {
                "copy_move": round(copy_move_score, 3),
                "high_frequency_ratio": round(high_freq_ratio, 3),
                "noise_inconsistency": round(float(max(diffs, peak_jump)), 3),
                "srm_residual": round(srm_score, 3),
            },
            "causal_chain": {
                "original": image_payload["id"],
                "manipulation_type": "splice/copy-move" if manipulated else "clean",
                "tool_hypothesis": "editor_or_generator" if manipulated else "unknown",
                "artifacts": reasons,
                "causal_parents": list(causal_weights.keys()),
                "causal_weights": causal_weights,
                "discovered_edges": [
                    {"source": edge.source, "target": edge.target, "strength": edge.strength} for edge in causal_edges
                ],
                "steps": [
                    "extract_residual_features",
                    "estimate_copy_move_consistency",
                    "score_frequency_anomalies",
                    "run_pc_discovery_on_patch_features",
                    "infer_manipulation_chain",
                ],
            },
        }

    def analyze_video(self, video_payload: dict) -> dict:
        frames = [np.array(frame) for frame in video_payload["frames"]]
        duplicates = sum(1 for idx in range(1, len(frames)) if np.array_equal(frames[idx - 1], frames[idx]))
        diffs = [float(np.mean(np.abs(frames[idx] - frames[idx - 1]))) for idx in range(1, len(frames))]
        temporal_inconsistency = float(np.std(diffs)) if diffs else 0.0
        compression = 0.0
        if cv2 is not None and frames:
            frame_float = frames[0].astype(np.float32)
            compression = float(np.mean(np.abs(cv2.dct(frame_float))))
        return {
            "asset_id": video_payload["id"],
            "frame_duplications": int(duplicates),
            "temporal_inconsistency": round(temporal_inconsistency, 3),
            "compression_artifacts": round(compression, 3),
            "manipulated": bool(duplicates > 0 or temporal_inconsistency > 0.6),
        }

    def analyze_audio(self, audio_payload: dict) -> dict:
        samples = np.array(audio_payload["samples"], dtype=np.float32)
        jumps = np.abs(np.diff(samples))
        splice_score = float(jumps.max()) if jumps.size else 0.0
        spectrum = np.abs(np.fft.rfft(samples))
        spectral_flux = float(np.mean(np.abs(np.diff(spectrum)))) if spectrum.size > 1 else 0.0
        background_score = float(np.std(samples[: len(samples) // 2]) + np.std(samples[len(samples) // 2 :]))
        return {
            "asset_id": audio_payload["id"],
            "splice_score": round(splice_score, 3),
            "spectral_flux": round(spectral_flux, 3),
            "background_inconsistency": round(background_score, 3),
            "manipulated": bool(splice_score > 0.4 or background_score > 0.3),
        }

    def batch_analyze_images(self, payloads: list[dict]) -> dict:
        if not payloads:
            return {"count": 0, "device": self.device, "manipulated": 0, "throughput_per_sec": 0.0}
        started = torch.cuda.Event(enable_timing=True) if torch is not None and self.device == "cuda" else None
        ended = torch.cuda.Event(enable_timing=True) if torch is not None and self.device == "cuda" else None
        if started is not None and ended is not None:
            started.record()
        else:
            import time

            wall_start = time.perf_counter()
        results = [self.analyze_image(payload) for payload in payloads]
        if started is not None and ended is not None:
            ended.record()
            torch.cuda.synchronize()
            elapsed_ms = float(started.elapsed_time(ended))
        else:
            import time

            elapsed_ms = float((time.perf_counter() - wall_start) * 1000)
        manipulated = sum(1 for item in results if item["manipulated"])
        throughput = (len(payloads) / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0
        return {
            "count": len(payloads),
            "device": self.device,
            "manipulated": manipulated,
            "throughput_per_sec": round(throughput, 3),
            "elapsed_ms": round(elapsed_ms, 3),
        }


def load_media(path: Path) -> dict:
    return json.loads(path.read_text())
