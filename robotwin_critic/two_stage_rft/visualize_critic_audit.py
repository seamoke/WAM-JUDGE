"""Build a human-reviewable HTML audit of Process and Action critic scores."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from robotwin_critic.two_stage_rft.kinematic_action_critic import (
    _prepend_condition_reference,
    kinematic_series,
)
from robotwin_critic.vlac_finetune.common import DEFAULT_CAMERAS, make_tshape_state


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(collect_root: Path) -> list[dict]:
    rows: list[dict] = []
    for dual_path in sorted(collect_root.glob("collect_*/dual_scored.jsonl")):
        collect_dir = dual_path.parent
        action_path = collect_dir / "action_scored_audit.jsonl"
        if not action_path.is_file():
            continue
        action_rows = {
            row["candidate_id"]: row["action_critic"]
            for row in read_jsonl(action_path)
        }
        for row in read_jsonl(dual_path):
            action = action_rows.get(row["candidate_id"])
            if action is None:
                continue
            row["action_critic"] = action
            row["collect"] = collect_dir.name
            rows.append(row)
    return rows


def diverse(rows: list[dict], count: int) -> list[dict]:
    selected: list[dict] = []
    seen_candidates: set[str] = set()
    seen_tasks: set[str] = set()
    seen_contexts: set[str] = set()
    for prefer_new_task in (True, False):
        for row in rows:
            candidate = str(row["candidate_id"])
            context = str(row["context_id"])
            task = str(row["task"])
            if candidate in seen_candidates or context in seen_contexts:
                continue
            if prefer_new_task and task in seen_tasks:
                continue
            selected.append(row)
            seen_candidates.add(candidate)
            seen_contexts.add(context)
            seen_tasks.add(task)
            if len(selected) >= count:
                return selected
    return selected


def sample_categories(rows: list[dict], count: int) -> list[tuple[str, str, list[dict]]]:
    numeric = [
        row
        for row in rows
        if row.get("process_critic", {}).get("numeric_parsed", False)
    ]
    action_high = sorted(
        numeric, key=lambda row: float(row["action_critic"]["action_score"]), reverse=True
    )
    action_low = list(reversed(action_high))
    process_high = sorted(numeric, key=lambda row: float(row["process_score"]), reverse=True)
    process_low = list(reversed(process_high))
    high_process_low_action = sorted(
        [
            row
            for row in numeric
            if float(row["process_score"]) >= 5.0
            and not bool(row["action_critic"]["accepted"])
        ],
        key=lambda row: (
            -float(row["process_score"]),
            float(row["action_critic"]["action_score"]),
        ),
    )
    low_process_high_action = sorted(
        [
            row
            for row in numeric
            if float(row["process_score"]) < 5.0
            and bool(row["action_critic"]["accepted"])
        ],
        key=lambda row: (
            float(row["process_score"]),
            -float(row["action_critic"]["action_score"]),
        ),
    )
    return [
        ("process-high", "Highest Process Critic scores", diverse(process_high, count)),
        ("process-low", "Lowest Process Critic scores", diverse(process_low, count)),
        ("action-high", "Highest Action Critic scores", diverse(action_high, count)),
        ("action-low", "Lowest Action Critic scores", diverse(action_low, count)),
        (
            "process-high-action-reject",
            "Process accepts, Action rejects",
            diverse(high_process_low_action, count),
        ),
        (
            "process-low-action-accept",
            "Process rejects, Action accepts",
            diverse(low_process_high_action, count),
        ),
    ]


def save_image(source: Path, target: Path, max_width: int = 640) -> None:
    image = Image.open(source).convert("RGB")
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    image.save(target, quality=90)


def save_cam_high_crop(source: Path, target: Path) -> None:
    """Keep the lower two-thirds occupied by cam_high in a T-shaped state."""
    image = Image.open(source).convert("RGB")
    top = image.height // 3
    image.crop((0, top, image.width, image.height)).save(target, quality=90)


def decode_video_frame(video: Path, frame: int, target: Path) -> tuple[np.ndarray | None, str]:
    """Decode one exact video frame without modifying the source video."""
    result = subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-frames:v",
            "1",
            str(target),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not target.is_file():
        return None, result.stderr[-300:]
    return np.asarray(Image.open(target).convert("RGB")), ""


def save_real_future(row: dict, target: Path) -> int:
    frame = min(
        int(row["frame_index"]) + int(row.get("executable_action_steps", 16)),
        int(row["length"]) - 1,
    )
    images: list[np.ndarray] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="robotwin-future-") as temp_dir:
        for camera in DEFAULT_CAMERAS:
            key = camera
            video_path = row.get("video_paths", {}).get(key)
            if not video_path:
                errors.append(f"missing {key}")
                continue
            decoded, error = decode_video_frame(
                Path(video_path), frame, Path(temp_dir) / f"{camera.rsplit('.', 1)[-1]}.png"
            )
            if decoded is None:
                errors.append(f"{camera}: {error}")
            else:
                images.append(decoded)
    if len(images) == len(DEFAULT_CAMERAS):
        current_width = Image.open(row["current_image"]).width
        mosaic = make_tshape_state(images, output_width=current_width)
        Image.fromarray(mosaic).save(target, quality=90)
    else:
        placeholder = Image.new("RGB", (640, 360), "#e2e8f0")
        draw = ImageDraw.Draw(placeholder)
        draw.text((20, 20), f"Future frame {frame} decode failed", fill="#991b1b")
        draw.text((20, 50), " | ".join(errors)[-500:], fill="#334155")
        placeholder.save(target, quality=90)
    return frame


def save_action_plot(row: dict, target: Path) -> None:
    action = np.load(row["action_path"])
    start = np.asarray(row["start_state"], dtype=np.float64)
    series = kinematic_series(
        _prepend_condition_reference(action, start_state=start), fps=30.0
    )
    diagnostics = row["action_critic"]["diagnostics"]
    metrics = (
        ("linear_velocity", "Linear velocity"),
        ("linear_acceleration", "Linear acceleration"),
        ("linear_jerk", "Linear jerk"),
        ("gripper_velocity", "Gripper velocity"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for arm, color in (("left", "#0f766e"), ("right", "#c2410c")):
            key = f"{arm}.{metric}"
            values = series[key]
            axis.plot(np.arange(len(values)), values, label=arm, color=color, linewidth=1.8)
            hard = float(diagnostics[key]["hard"])
            axis.axhline(hard, color=color, linestyle="--", alpha=0.45, linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Derivative step")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(loc="upper right")
    figure.suptitle(
        f"action={row['action_critic']['action_score']:.3f} | "
        f"process={float(row['process_score']):.1f}",
        fontsize=12,
    )
    figure.savefig(target, dpi=130)
    plt.close(figure)


def diagnostic_text(row: dict) -> str:
    critic = row["action_critic"]
    diagnostics = critic.get("diagnostics", {})
    ranked = sorted(
        diagnostics.items(),
        key=lambda item: float(item[1].get("maximum", 0.0))
        / max(float(item[1].get("hard", 0.0)), 1e-12),
        reverse=True,
    )[:4]
    ratios = ", ".join(
        f"{name}={float(value['maximum']) / max(float(value['hard']), 1e-12):.2f}x"
        for name, value in ranked
    )
    gates = critic.get("gate_violations", [])
    return f"gate={gates or 'none'}; largest max/hard: {ratios}"


def save_summary_plot(rows: list[dict], target: Path) -> None:
    process = np.asarray([float(row["process_score"]) for row in rows])
    action = np.asarray([float(row["action_critic"]["action_score"]) for row in rows])
    accepted = np.asarray(
        [
            bool(row["action_critic"]["accepted"])
            and float(row["process_score"]) >= 5.0
            for row in rows
        ]
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].hist(process, bins=30, color="#2563eb", alpha=0.85)
    axes[0].axvline(5.0, color="#b91c1c", linestyle="--", label="threshold=5")
    axes[0].set_title("Process score distribution")
    axes[0].legend()
    axes[1].hist(action, bins=30, color="#0f766e", alpha=0.85)
    axes[1].axvline(0.75, color="#b91c1c", linestyle="--", label="threshold=0.75")
    axes[1].set_title("Action score distribution")
    axes[1].legend()
    axes[2].scatter(action[~accepted], process[~accepted], s=8, alpha=0.3, color="#64748b")
    axes[2].scatter(action[accepted], process[accepted], s=12, alpha=0.7, color="#16a34a")
    axes[2].axvline(0.75, color="#b91c1c", linestyle="--", linewidth=1)
    axes[2].axhline(5.0, color="#b91c1c", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Action score")
    axes[2].set_ylabel("Process score")
    axes[2].set_title("Joint critic decisions")
    figure.savefig(target, dpi=140)
    plt.close(figure)


def build_overview(samples: list[dict], output: Path) -> None:
    chosen = samples[:8]
    tile = (300, 225)
    canvas = Image.new("RGB", (tile[0] * 3, (tile[1] + 42) * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, sample in enumerate(chosen):
        y = index * (tile[1] + 42)
        draw.text(
            (6, y + 4),
            f"{sample['category']} | {sample['task']} | P={sample['process_score']:.1f} A={sample['action_score']:.3f}",
            fill="#111827",
        )
        for column, key in enumerate(("current_asset", "future_asset", "generated_asset")):
            image = Image.open(output.parent / sample[key]).convert("RGB")
            image.thumbnail(tile, Image.Resampling.LANCZOS)
            x = column * tile[0] + (tile[0] - image.width) // 2
            canvas.paste(image, (x, y + 42 + (tile[1] - image.height) // 2))
    canvas.save(output, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.collect_root)
    if not rows:
        raise RuntimeError("No completed dual/action audit pairs were found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    categories = sample_categories(rows, args.per_category)
    sampled: list[dict] = []
    sections: list[str] = []
    sample_index = 0
    for category, title, selected in categories:
        cards: list[str] = []
        for row in selected:
            stem = f"sample_{sample_index:03d}"
            current = assets / f"{stem}_current.jpg"
            future = assets / f"{stem}_real_future.jpg"
            generated = assets / f"{stem}_generated.jpg"
            current_high = assets / f"{stem}_current_cam_high.jpg"
            future_high = assets / f"{stem}_real_future_cam_high.jpg"
            generated_high = assets / f"{stem}_generated_cam_high.jpg"
            action_plot = assets / f"{stem}_action.png"
            save_image(Path(row["current_image"]), current)
            save_image(Path(row["generated_image"]), generated)
            future_frame = save_real_future(row, future)
            save_cam_high_crop(current, current_high)
            save_cam_high_crop(future, future_high)
            save_cam_high_crop(generated, generated_high)
            save_action_plot(row, action_plot)
            record = {
                "category": category,
                "candidate_id": row["candidate_id"],
                "context_id": row["context_id"],
                "task": row["task"],
                "domain": row["domain"],
                "instruction": row["text"],
                "process_score": float(row["process_score"]),
                "original_process_score": row.get("original_process_score"),
                "process_raw": row.get("process_critic", {}).get("raw_answer"),
                "action_score": float(row["action_critic"]["action_score"]),
                "action_accepted": bool(row["action_critic"]["accepted"]),
                "gate_violations": row["action_critic"].get("gate_violations", []),
                "hard_violations": row["action_critic"].get("hard_violations", []),
                "future_frame": future_frame,
                "source_frame": int(row["frame_index"]),
                "source_action_path": row["action_path"],
                "current_asset": str(current.relative_to(args.output_dir)),
                "future_asset": str(future.relative_to(args.output_dir)),
                "generated_asset": str(generated.relative_to(args.output_dir)),
                "current_cam_high_asset": str(current_high.relative_to(args.output_dir)),
                "future_cam_high_asset": str(future_high.relative_to(args.output_dir)),
                "generated_cam_high_asset": str(generated_high.relative_to(args.output_dir)),
                "action_asset": str(action_plot.relative_to(args.output_dir)),
            }
            sampled.append(record)
            cards.append(
                f"""
                <article class="sample">
                  <h3>{html.escape(str(row['task']))} <span>{html.escape(str(row['domain']))}</span></h3>
                  <p class="instruction">{html.escape(str(row['text']))}</p>
                  <div class="scores"><b>Process {float(row['process_score']):.1f}</b>{f"<b>Old black-image Process {float(row['original_process_score']):.1f}</b>" if row.get('original_process_score') is not None else ''}<b>Action {float(row['action_critic']['action_score']):.3f}</b><b>Action accepted: {bool(row['action_critic']['accepted'])}</b></div>
                  <div class="frames">
                    <figure><img src="{current.relative_to(args.output_dir)}?v=tshape3"><figcaption>Real current t={row['frame_index']} (3-camera T mosaic)</figcaption></figure>
                    <figure><img src="{future.relative_to(args.output_dir)}?v=tshape3"><figcaption>Real future t={future_frame} (same mosaic)</figcaption></figure>
                    <figure><img src="{generated.relative_to(args.output_dir)}?v=tshape3"><figcaption>WAM generated future (same camera layout)</figcaption></figure>
                  </div>
                  <h4>cam_high-only crop</h4>
                  <div class="frames high-only">
                    <figure><img src="{current_high.relative_to(args.output_dir)}?v=tshape3"><figcaption>Real current cam_high</figcaption></figure>
                    <figure><img src="{future_high.relative_to(args.output_dir)}?v=tshape3"><figcaption>Real future cam_high</figcaption></figure>
                    <figure><img src="{generated_high.relative_to(args.output_dir)}?v=tshape3"><figcaption>WAM generated cam_high</figcaption></figure>
                  </div>
                  <img class="plot" src="{action_plot.relative_to(args.output_dir)}">
                  <p class="diagnostic">{html.escape(diagnostic_text(row))}</p>
                  <details><summary>Candidate identity and raw Process output</summary><pre>{html.escape(json.dumps({'candidate_id': row['candidate_id'], 'context_id': row['context_id'], 'process_raw': row.get('process_critic', {}).get('raw_answer'), 'hard_violations': row['action_critic'].get('hard_violations', [])}, indent=2))}</pre></details>
                </article>
                """
            )
            sample_index += 1
        sections.append(f"<section><h2 id='{category}'>{html.escape(title)}</h2>{''.join(cards)}</section>")

    numeric = [row for row in rows if row.get("process_critic", {}).get("numeric_parsed")]
    action_pass = sum(bool(row["action_critic"]["accepted"]) for row in numeric)
    process_pass = sum(float(row["process_score"]) >= 5.0 for row in numeric)
    dual_pass = sum(
        bool(row["action_critic"]["accepted"]) and float(row["process_score"]) >= 5.0
        for row in numeric
    )
    summary = {
        "candidates": len(numeric),
        "contexts": len({row["context_id"] for row in numeric}),
        "tasks": len({row["task"] for row in numeric}),
        "action_pass": action_pass,
        "action_pass_rate": action_pass / len(numeric),
        "process_pass": process_pass,
        "process_pass_rate": process_pass / len(numeric),
        "dual_pass": dual_pass,
        "dual_pass_rate": dual_pass / len(numeric),
        "process_score_counts": dict(Counter(float(row["process_score"]) for row in numeric).most_common()),
    }
    save_summary_plot(numeric, assets / "score_summary.png")
    (args.output_dir / "samples.json").write_text(
        json.dumps(sampled, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_overview(sampled, args.output_dir / "overview.jpg")
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RoboTwin Critic Audit</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;color:#111827;background:#f8fafc;line-height:1.45}}main{{max-width:1440px;margin:auto;padding:28px}}h1{{font-size:30px}}h2{{margin-top:42px;border-bottom:2px solid #cbd5e1;padding-bottom:8px}}h3 span{{font-size:13px;font-weight:normal;color:#475569}}h4{{margin:18px 0 8px;color:#334155}}.summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.metric{{background:white;border:1px solid #cbd5e1;padding:14px;border-radius:6px}}.metric b{{display:block;font-size:25px}}.summaryplot{{width:100%;margin-top:16px;background:white}}.sample{{background:white;border:1px solid #cbd5e1;border-radius:6px;padding:16px;margin:16px 0}}.instruction{{color:#334155}}.scores{{display:flex;gap:24px;padding:8px 0}}.frames{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.high-only{{padding-top:8px;border-top:1px solid #e2e8f0}}figure{{margin:0}}figure img{{width:100%;height:260px;object-fit:contain;background:#0f172a}}figcaption{{text-align:center;padding:5px;color:#475569}}.plot{{display:block;width:min(100%,1000px);margin:12px auto}}.diagnostic{{font-family:monospace;overflow-wrap:anywhere}}pre{{white-space:pre-wrap}}@media(max-width:800px){{.summary,.frames{{grid-template-columns:1fr}}.scores{{flex-direction:column;gap:4px}}}}
</style></head><body><main>
<h1>RoboTwin Chunk Critic Human Audit</h1>
<p>The first row uses the same T-shaped camera layout for all states: left/right wrist cameras on top and cam_high below. The second row crops every image to cam_high only for visual diagnosis; displayed Process scores still come from the original three-camera VLAC input. Dashed action-plot lines are Stage-1-calibrated hard thresholds.</p>
<div class="summary">
<div class="metric">Candidates<b>{summary['candidates']}</b></div>
<div class="metric">Action pass<b>{summary['action_pass_rate']:.1%}</b></div>
<div class="metric">Process pass<b>{summary['process_pass_rate']:.1%}</b></div>
<div class="metric">Dual pass<b>{summary['dual_pass_rate']:.1%}</b></div>
</div><img class="summaryplot" src="assets/score_summary.png">
{''.join(sections)}
</main></body></html>"""
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps({"summary": summary, "samples": len(sampled), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
