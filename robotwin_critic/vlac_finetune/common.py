from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from scipy.stats import spearmanr


DEFAULT_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

SYSTEM_PROMPT = (
    "You are a visual-language assistant designed to interpret spatial and task-related "
    "information from images and text. Provide precise, context-aware responses and "
    "actionable guidance to assist in achieving task objectives."
)

PAIR_PROMPT = (
    "Image-1: <image>\n"
    "Image-2: <image>\n"
    "Compare two images and evaluate whether the second image is closer to achieving "
    "task objectives compared to the first image. + score means the second image is "
    "closer, - score means the first image is closer. Respond with only one signed "
    "numeric progress score in percent. The target task is: <task> {task} </task> <score>"
)

SCORE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


class VideoDecodeError(RuntimeError):
    """Raised when neither PyAV nor OpenCV can decode a video."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def stable_fraction(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def parse_score(text: str, clip: float = 100.0) -> float:
    score_section = text
    if "<score>" in text:
        score_section = text.rsplit("<score>", 1)[-1]
    match = SCORE_RE.search(score_section)
    if not match:
        match = SCORE_RE.search(text)
    if not match:
        raise ValueError(f"VLAC response has no numeric score: {text!r}")
    return float(np.clip(float(match.group()), -clip, clip))


def format_score(score: float) -> str:
    score = float(np.clip(score, -100.0, 100.0))
    if abs(score) < 0.05:
        return "0"
    return f"{score:+.1f}".rstrip("0").rstrip(".")


def pair_prompt(task: str) -> str:
    return PAIR_PROMPT.format(task=task)


def accumulate_progress(scores_percent: Sequence[float]) -> np.ndarray:
    values = [0.0]
    for score in scores_percent:
        fraction = float(score) / 100.0
        values.append(values[-1] + fraction * (1.0 - values[-1]))
    return np.asarray(values, dtype=np.float64)


def spearman_order(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.allclose(values, values[0]):
        return 0.0
    time_order = np.arange(len(values), dtype=float)
    corr = spearmanr(values, time_order).statistic
    return 0.0 if not np.isfinite(corr) else float(corr)


def voc_f1(voc: float, vroc: float) -> float:
    if voc <= 0 or vroc <= 0 or math.isclose(voc + vroc, 0.0):
        return 0.0
    return float(2.0 * voc * vroc / (voc + vroc))


class VideoFrameReader:
    def __init__(self, max_cached_videos: int = 3):
        self.max_cached_videos = int(max_cached_videos)
        self._frames: OrderedDict[str, list[np.ndarray]] = OrderedDict()

    @staticmethod
    def _decode_all(path: str) -> list[np.ndarray]:
        def decode_with_opencv() -> list[np.ndarray]:
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise FileNotFoundError(f"Cannot open video: {path}")
            decoded: list[np.ndarray] = []
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            capture.release()
            return decoded

        try:
            import av
        except ImportError:
            frames = decode_with_opencv()
        else:
            try:
                frames = []
                with av.open(path) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    for frame in container.decode(stream):
                        frames.append(frame.to_ndarray(format="rgb24"))
            except Exception:
                # Some RoboTwin MP4s contain packets PyAV rejects but OpenCV can decode.
                frames = decode_with_opencv()
        if not frames:
            raise VideoDecodeError(f"Video decoded zero frames: {path}")
        return frames

    def read(self, path: str | Path, frame_index: int) -> np.ndarray:
        key = str(path)
        frames = self._frames.pop(key, None)
        if frames is None:
            if not Path(key).exists():
                raise FileNotFoundError(f"Cannot open video: {key}")
            frames = self._decode_all(key)
        self._frames[key] = frames
        while len(self._frames) > self.max_cached_videos:
            self._frames.popitem(last=False)
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= len(frames):
            raise IndexError(
                f"Cannot read frame {frame_index} from {key}; decoded {len(frames)} frames"
            )
        return frames[frame_index]

    def close(self):
        self._frames.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def make_tshape_state(images: Sequence[np.ndarray], output_width: int = 448) -> np.ndarray:
    if len(images) != 3:
        raise ValueError("T-shape state requires high, left-wrist, and right-wrist RGB images")
    output_width = int(output_width)
    wrist_height = output_width // 3
    high_height = output_width - wrist_height
    wrist_width = output_width // 2
    high = cv2.resize(images[0], (output_width, high_height), interpolation=cv2.INTER_AREA)
    wrists = [
        cv2.resize(image, (wrist_width, wrist_height), interpolation=cv2.INTER_AREA)
        for image in images[1:]
    ]
    return np.concatenate([np.concatenate(wrists, axis=1), high], axis=0)


def normalized_pixel_difference(first: np.ndarray, second: np.ndarray) -> float:
    first_small = cv2.resize(first, (112, 168), interpolation=cv2.INTER_AREA).astype(np.float32)
    second_small = cv2.resize(second, (112, 168), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.mean(np.abs(first_small - second_small)) / 255.0)
