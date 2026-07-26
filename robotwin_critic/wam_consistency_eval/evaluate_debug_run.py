"""Decode WAM debug latents and compare them with executed RoboTwin observations.

The existing WAM server already saves ``latents_<frame>.pt`` during inference and
``obs_data_<frame>.pt`` after the corresponding action chunk is executed when
``WAN_VA_SAVE_INFER_DEBUG=1``. This module consumes those artifacts without
changing the WAM server or RoboTwin client.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from diffusers.video_processor import VideoProcessor

try:
    from skimage.metrics import structural_similarity
except ImportError as exc:  # pragma: no cover - environment validation handles this
    raise RuntimeError("scikit-image is required for SSIM evaluation") from exc


FRAME_RE = re.compile(r"(?:latents|obs_data)_(\d+)\.pt$")


@dataclass
class PairRecord:
    run: str
    chunk_start: int
    pred_index: int
    real_index: int
    pair_type: str
    is_match: int
    mae: float
    psnr: float
    ssim: float
    feature_cosine: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", action="append", required=True)
    parser.add_argument("--project-root", default="/workspace/lingbot-va")
    parser.add_argument("--config-name", default="robotwin")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--feature-model", default=None)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def _load_project_components(project_root: Path, config_name: str):
    wan_dir = project_root / "wan_va"
    if str(wan_dir) not in sys.path:
        sys.path.insert(0, str(wan_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.utils import load_vae  # type: ignore
    from wan_va.configs import VA_CONFIGS  # type: ignore

    if config_name not in VA_CONFIGS:
        raise KeyError(f"Unknown WAM config: {config_name}")
    return VA_CONFIGS[config_name], load_vae


def _torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _frame_id(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse frame id from {path}")
    return int(match.group(1))


def discover_chunks(debug_dir: Path) -> list[tuple[int, Path, Path]]:
    latents = {_frame_id(path): path for path in debug_dir.glob("latents_*.pt")}
    observations = {_frame_id(path): path for path in debug_dir.glob("obs_data_*.pt")}
    shared = sorted(set(latents).intersection(observations))
    return [(frame, latents[frame], observations[frame]) for frame in shared]


def _as_rgb_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def observation_mosaic(
    observation: dict[str, Any],
    camera_keys: Sequence[str],
    env_type: str,
    width: int,
    height: int,
) -> np.ndarray:
    missing = [key for key in camera_keys if key not in observation]
    if missing:
        raise KeyError(f"Observation is missing cameras: {missing}")

    images = [_as_rgb_uint8(observation[key]) for key in camera_keys]
    if env_type == "robotwin_tshape":
        if len(images) != 3:
            raise ValueError("robotwin_tshape requires exactly three cameras")
        high = _resize_rgb(images[0], width, height)
        wrists = [_resize_rgb(image, width // 2, height // 2) for image in images[1:]]
        return np.concatenate([np.concatenate(wrists, axis=1), high], axis=0)

    resized = [_resize_rgb(image, width, height) for image in images]
    return np.concatenate(resized, axis=1)


def unpack_observations(payload: Any) -> list[dict[str, Any]]:
    while isinstance(payload, dict) and "obs" in payload:
        payload = payload["obs"]
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.tolist()
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"Unsupported observation payload: {type(payload)}")
    observations = list(payload)
    if not all(isinstance(item, dict) for item in observations):
        raise TypeError("Observation payload must be a list of camera dictionaries")
    return observations


def load_vae_decoder(load_vae, base_model: Path, dtype: torch.dtype, device: torch.device):
    vae_path = base_model / "vae"
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE directory does not exist: {vae_path}")
    vae = load_vae(str(vae_path), torch_dtype=dtype, torch_device=device)
    vae.eval()
    processor = VideoProcessor(vae_scale_factor=1)
    return vae, processor


@torch.no_grad()
def decode_latents(
    latents: torch.Tensor,
    vae: torch.nn.Module,
    processor: VideoProcessor,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    latents = latents.to(device=device, dtype=dtype)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    inverse_std = 1.0 / torch.tensor(
        vae.config.latents_std, device=device, dtype=dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    decoded = vae.decode(latents / inverse_std + mean, return_dict=False)[0]
    video = processor.postprocess_video(decoded, output_type="np")[0]
    return np.stack([_as_rgb_uint8(frame) for frame in video])


def frame_metrics(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    if predicted.shape != actual.shape:
        actual = _resize_rgb(actual, predicted.shape[1], predicted.shape[0])
    pred_float = predicted.astype(np.float32) / 255.0
    real_float = actual.astype(np.float32) / 255.0
    mae = float(np.mean(np.abs(pred_float - real_float)))
    mse = float(np.mean((pred_float - real_float) ** 2))
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))
    ssim = float(structural_similarity(predicted, actual, channel_axis=2, data_range=255))
    return mae, psnr, ssim


def _rank_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positive = labels_array == 1
    negative = labels_array == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    order = np.argsort(scores_array, kind="mergesort")
    ranks = np.empty(len(scores_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores_array[order[end]] == scores_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _median_gap(records: Sequence[PairRecord], field: str) -> float:
    matched = [getattr(item, field) for item in records if item.is_match]
    mismatched = [getattr(item, field) for item in records if not item.is_match]
    matched = [value for value in matched if value is not None]
    mismatched = [value for value in mismatched if value is not None]
    if not matched or not mismatched:
        return float("nan")
    return float(np.median(matched) - np.median(mismatched))


class FeatureEncoder:
    def __init__(self, model_path: str, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).eval().to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, images: Sequence[np.ndarray], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            result = self.model(**inputs)
            if getattr(result, "image_embeds", None) is not None:
                embedding = result.image_embeds
            elif getattr(result, "pooler_output", None) is not None:
                embedding = result.pooler_output
            else:
                embedding = result.last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
            outputs.append(embedding.cpu().numpy())
        return np.concatenate(outputs, axis=0)


def write_video(path: Path, predicted: Sequence[np.ndarray], actual: Sequence[np.ndarray], fps: float):
    if not predicted:
        return
    height = max(predicted[0].shape[0], actual[0].shape[0])
    width = predicted[0].shape[1] + actual[0].shape[1]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Cannot open video writer for {path}")
    try:
        for pred, real in zip(predicted, actual):
            if pred.shape[:2] != (height, pred.shape[1]):
                pred = _resize_rgb(pred, pred.shape[1], height)
            if real.shape[:2] != (height, real.shape[1]):
                real = _resize_rgb(real, real.shape[1], height)
            frame = np.concatenate([pred, real], axis=1)
            cv2.putText(frame, "WAM prediction", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(
                frame,
                "RoboTwin actual",
                (pred.shape[1] + 12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def mismatch_indices(length: int, index: int) -> Iterable[tuple[str, int]]:
    if length < 2:
        return
    candidates = {
        "half_shift": (index + max(1, length // 2)) % length,
        "reverse": length - 1 - index,
    }
    for name, other in candidates.items():
        if other != index:
            yield name, other


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, load_vae = _load_project_components(project_root, args.config_name)
    base_model = Path(args.base_model or config.wan22_pretrained_model_name_or_path)
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else config.param_dtype
    vae, processor = load_vae_decoder(load_vae, base_model, dtype, device)

    records: list[PairRecord] = []
    feature_images_a: list[np.ndarray] = []
    feature_images_b: list[np.ndarray] = []
    decoded_chunks: list[tuple[str, int, list[np.ndarray], list[np.ndarray]]] = []

    for debug_dir_text in args.debug_dir:
        debug_dir = Path(debug_dir_text)
        chunks = discover_chunks(debug_dir)
        if args.chunk_offset > 0:
            chunks = chunks[args.chunk_offset :]
        if args.max_chunks > 0:
            chunks = chunks[: args.max_chunks]
        if not chunks:
            raise FileNotFoundError(f"No aligned latent/observation chunks in {debug_dir}")

        for chunk_start, latent_path, obs_path in chunks:
            latent = _torch_load(latent_path)
            if isinstance(latent, dict):
                latent = latent.get("latents", latent.get("latent"))
            if not isinstance(latent, torch.Tensor):
                raise TypeError(f"No latent tensor in {latent_path}")
            predicted = decode_latents(latent, vae, processor, device, dtype)
            observations = unpack_observations(_torch_load(obs_path))
            actual = [
                observation_mosaic(
                    item,
                    config.obs_cam_keys,
                    config.env_type,
                    config.width,
                    config.height,
                )
                for item in observations
            ]

            if chunk_start == 0 and len(predicted) == len(actual) + 1:
                predicted = predicted[1:]
            pair_count = min(len(predicted), len(actual))
            predicted = list(predicted[:pair_count])
            actual = actual[:pair_count]
            decoded_chunks.append((debug_dir.name, chunk_start, predicted, actual))

            for pred_index in range(pair_count):
                candidates = [("matched", pred_index, 1)]
                candidates.extend(
                    (name, other_index, 0)
                    for name, other_index in mismatch_indices(pair_count, pred_index)
                )
                for pair_type, real_index, is_match in candidates:
                    mae, psnr, ssim = frame_metrics(predicted[pred_index], actual[real_index])
                    records.append(
                        PairRecord(
                            run=debug_dir.name,
                            chunk_start=chunk_start,
                            pred_index=pred_index,
                            real_index=real_index,
                            pair_type=pair_type,
                            is_match=is_match,
                            mae=mae,
                            psnr=psnr,
                            ssim=ssim,
                        )
                    )
                    feature_images_a.append(predicted[pred_index])
                    feature_images_b.append(actual[real_index])

            torch.cuda.empty_cache()

    if args.feature_model:
        encoder = FeatureEncoder(args.feature_model, device)
        features_a = encoder.encode(feature_images_a, args.feature_batch_size)
        features_b = encoder.encode(feature_images_b, args.feature_batch_size)
        similarities = np.sum(features_a * features_b, axis=1)
        for record, similarity in zip(records, similarities):
            record.feature_cosine = float(similarity)

    if not records:
        raise RuntimeError("Aligned debug chunks contained no comparable frames")

    for run, chunk_start, predicted, actual in decoded_chunks:
        write_video(output_dir / f"{run}_chunk_{chunk_start}.mp4", predicted, actual, args.fps)

    fieldnames = list(records[0].as_dict())
    with (output_dir / "frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.as_dict() for record in records)

    labels = [record.is_match for record in records]
    metric_specs = {"mae": -1.0, "psnr": 1.0, "ssim": 1.0}
    if args.feature_model:
        metric_specs["feature_cosine"] = 1.0
    metrics: dict[str, Any] = {}
    for field, direction in metric_specs.items():
        values = [getattr(record, field) for record in records]
        scores = [direction * float(value) for value in values if value is not None]
        valid_labels = [label for label, value in zip(labels, values) if value is not None]
        metrics[field] = {
            "auc_matched_vs_mismatch": _rank_auc(valid_labels, scores),
            "matched_minus_mismatch_median": _median_gap(records, field),
            "matched_median": float(
                np.median([float(value) for value, label in zip(values, labels) if label and value is not None])
            ),
            "mismatch_median": float(
                np.median([float(value) for value, label in zip(values, labels) if not label and value is not None])
            ),
        }

    sample_count = sum(labels)
    semantic = metrics.get("feature_cosine")
    if sample_count < 30:
        recommendation = "collect_more_aligned_chunks"
    elif semantic and semantic["auc_matched_vs_mismatch"] >= 0.75:
        recommendation = "consistency_filter_has_discriminative_signal"
    elif metrics["ssim"]["auc_matched_vs_mismatch"] >= 0.75:
        recommendation = "structural_signal_only_add_semantic_encoder_before_filtering"
    else:
        recommendation = "current_distances_do_not_support_a_reliable_filter"

    summary = {
        "runs": args.debug_dir,
        "chunk_offset": args.chunk_offset,
        "matched_frames": sample_count,
        "mismatch_pairs": len(records) - sample_count,
        "metrics": metrics,
        "recommendation": recommendation,
        "interpretation": (
            "AUC measures whether true WAM-to-simulator pairs are more similar than deliberately "
            "mismatched pairs. This establishes discriminability, not causal downstream benefit; "
            "the final filter threshold must also be validated against task success and false rejects."
        ),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
