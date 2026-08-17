"""Torch-free scheduling and exposure math for joint RFT."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

from robotwin_critic.two_stage_rft.model_completeness import (
    is_complete_transformer,
)


SCHEDULE_MODES = ("steps", "base-auxiliary-pseudo", "epochs")
TOTAL_STEP_DEPENDENT_SCHEDULERS = frozenset({"cosine"})
BASE_AUXILIARY_PRODUCTION_STEPS = 15000
BASE_AUXILIARY_PRODUCTION_SAVE_STEPS = [3000, 6000, 9000, 12000, 15000]


def resolve_base_auxiliary_steps(
    configured_steps: int,
    smoke_steps: int = 0,
    regression_steps: int = 0,
    configured_save_steps: list[int] | tuple[int, ...] | None = None,
) -> tuple[int, list[int]]:
    """Validate an arbitrary positive step count and its checkpoint schedule."""
    configured_steps = int(configured_steps)
    smoke_steps = int(smoke_steps)
    regression_steps = int(regression_steps)
    if smoke_steps and regression_steps:
        raise ValueError("smoke and real-only regression allowances are mutually exclusive")
    if smoke_steps:
        if not 1 <= smoke_steps <= 500:
            raise ValueError("base auxiliary smoke steps must be between 1 and 500")
        if configured_steps != smoke_steps:
            raise ValueError(
                "explicit base auxiliary smoke allowance must match config.num_steps"
            )
        return configured_steps, [smoke_steps]
    if regression_steps:
        if not 1 <= regression_steps <= 2_000:
            raise ValueError("real-only regression steps must be between 1 and 2000")
        if configured_steps != regression_steps:
            raise ValueError(
                "explicit real-only regression allowance must match config.num_steps"
            )
        return configured_steps, [regression_steps]
    if configured_steps <= 0:
        raise ValueError("base auxiliary optimizer steps must be positive")
    if configured_save_steps is None:
        save_steps = (
            list(BASE_AUXILIARY_PRODUCTION_SAVE_STEPS)
            if configured_steps == BASE_AUXILIARY_PRODUCTION_STEPS
            else [configured_steps]
        )
    else:
        save_steps = [int(step) for step in configured_save_steps]
        if not save_steps:
            raise ValueError("base auxiliary save steps must not be empty")
        if any(step <= 0 or step > configured_steps for step in save_steps):
            raise ValueError(
                "base auxiliary save steps must be positive and no greater than optimizer steps"
            )
        if save_steps != sorted(set(save_steps)):
            raise ValueError("base auxiliary save steps must be sorted and unique")
    return configured_steps, save_steps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path: str | Path) -> dict[str, object]:
    """Return stable initializer identity available without importing Torch."""
    model = Path(path).expanduser().resolve()
    transformer = model / "transformer"
    weight_files = sorted(transformer.glob("*.safetensors"))
    identity: dict[str, object] = {
        "canonical_path": str(model),
        "transformer_config_sha256": _sha256(transformer / "config.json"),
        "transformer_weight_sha256": {
            weight.name: _sha256(weight) for weight in weight_files
        },
    }
    return identity


def valid_base_auxiliary_completion(
    path: str | Path, *, pseudo_path: str | Path | None = None,
    pseudo_sampler_seed: int | None = None, target_weight: float | None = None,
    warmup_steps: int | None = None, schedule: str | None = None,
    initializer: str | Path | None = None,
) -> bool:
    """Return whether a completion marker proves the Base auxiliary run finished."""
    try:
        marker = json.loads(Path(path).read_text(encoding="utf-8"))
        report = marker["exposure_report"]
        steps = int(marker["optimizer_steps"])
        pseudo_global_batch = int(report["pseudo_global_batch"])
        pseudo_enabled = float(marker["pseudo_loss_weight_target"]) > 0
        final_model = Path(marker["final_model"])
        canonical_pseudo = Path(marker["pseudo_artifact"]["canonical_path"])
        pseudo_matches_bytes = (
            canonical_pseudo.is_file()
            and marker["pseudo_artifact"]["sha256"] == _sha256(canonical_pseudo)
        )
        expected_matches = all((
            pseudo_path is None or canonical_pseudo == Path(pseudo_path).expanduser().resolve(),
            pseudo_sampler_seed is None
            or int(marker["pseudo_sampler_seed"]) == int(pseudo_sampler_seed),
            target_weight is None
            or float(marker["pseudo_loss_weight_target"]) == float(target_weight),
            warmup_steps is None
            or int(marker["pseudo_loss_warmup_steps"]) == int(warmup_steps),
            schedule is None or marker["schedule"] == schedule,
            initializer is None
            or marker["initializer_identity"] == model_identity(initializer),
        ))
        return bool(
            marker.get("complete") is True
            and marker.get("schedule") == "base-auxiliary-pseudo"
            and isinstance(marker["pseudo_sampler_seed"], int)
            and float(marker["pseudo_loss_weight_target"]) >= 0
            and int(marker["pseudo_loss_warmup_steps"]) >= 0
            and marker["initializer_identity"]["canonical_path"]
            and len(marker["initializer_identity"]["transformer_config_sha256"]) == 64
            and marker["final_model_identity"] == model_identity(final_model)
            and pseudo_matches_bytes
            and expected_matches
            and steps > 0
            and int(report["optimizer_steps"]) == steps
            and int(report["observed_real_samples"]) == steps * 64
            and int(report["observed_pseudo_samples"])
            == (steps * pseudo_global_batch if pseudo_enabled else 0)
            and is_complete_transformer(final_model / "transformer")
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def effective_pseudo_loss_weight(
    *, target_weight: float, warmup_steps: int, optimizer_step: int,
) -> float:
    """Return the pseudo coefficient for a zero-based optimizer step."""
    if target_weight < 0:
        raise ValueError("target_weight must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    if warmup_steps == 0:
        return target_weight
    return target_weight * min((optimizer_step + 1) / warmup_steps, 1.0)


def auxiliary_update_plan(
    *, real_microbatches: int, pseudo_microbatches: int,
    pseudo_loss_weight: float,
) -> dict[str, object]:
    """Describe draws and the independent real/pseudo synchronization boundaries."""
    if real_microbatches <= 0 or pseudo_microbatches <= 0:
        raise ValueError("real and pseudo microbatch counts must be positive")
    if pseudo_loss_weight < 0:
        raise ValueError("pseudo_loss_weight must be non-negative")
    pseudo_enabled = pseudo_loss_weight > 0
    return {
        "pseudo_enabled": pseudo_enabled,
        "pseudo_draws_per_rank_update": pseudo_microbatches if pseudo_enabled else 0,
        # Preserve Base exactly: its final real microbatch is authoritative.
        "real_sync_flags": [
            index + 1 == real_microbatches for index in range(real_microbatches)
        ],
        "pseudo_sync_flags": (
            [index + 1 == pseudo_microbatches for index in range(pseudo_microbatches)]
            if pseudo_enabled
            else []
        ),
    }


def auxiliary_scaling_config(
    *, real_microbatches: int, pseudo_microbatches: int,
    pseudo_loss_weight: float,
) -> dict[str, object]:
    """Derive production accumulation/synchronization and pseudo scaling."""
    plan = auxiliary_update_plan(
        real_microbatches=real_microbatches,
        pseudo_microbatches=pseudo_microbatches,
        pseudo_loss_weight=pseudo_loss_weight,
    )
    return {
        "auxiliary_update_plan": plan,
        "pseudo_backward_scale_per_weight": (
            real_microbatches / pseudo_microbatches
        ),
    }


def next_sampler_batch(loader, iterator, sampler, epoch: int):
    """Return the next batch and deterministically advance sampler epochs."""
    if iterator is None:
        sampler.set_epoch(epoch)
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        epoch += 1
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator, epoch


def validate_base_auxiliary_contract(
    *, mixing_mode: str, real_chunk_mode: str, batch_size_per_rank: int,
    global_batch_size: int, activation_checkpointing: bool, warmup_steps: int,
    scheduler_type: str, max_episode_frames: int, num_epochs: int,
    real_data_mode: str, real_data_root: str | Path,
    prepared_data_root: str | Path,
    required_activation_checkpointing: bool = False,
) -> None:
    """Validate Base recipe invariants for direct trainer invocation."""
    required_real_mode = "stage1"
    required_real_root = (
        Path(prepared_data_root).expanduser().resolve() / "stage1"
    ).resolve()
    if real_data_mode in ("stage2", "stage1-stage2"):
        manifest_path = (
            Path(prepared_data_root).expanduser().resolve() / "split_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_real_mode = real_data_mode
        required_real_root = Path(manifest["source_root"]).expanduser().resolve()
    expected = {
        "mixing_mode": (mixing_mode, "auxiliary"),
        "real_chunk_mode": (real_chunk_mode, "full"),
        "batch_size_per_rank": (batch_size_per_rank, 1),
        "global_batch_size": (global_batch_size, 64),
        "activation_checkpointing": (
            bool(activation_checkpointing),
            bool(required_activation_checkpointing),
        ),
        "warmup_steps": (warmup_steps, 10),
        "scheduler_type": (scheduler_type.strip().lower(), "constant"),
        "max_episode_frames": (max_episode_frames, 1_000_000_000),
        "num_epochs": (num_epochs, 0),
        "real_data_mode": (real_data_mode, required_real_mode),
        "real_data_root": (
            Path(real_data_root).expanduser().resolve(),
            required_real_root,
        ),
    }
    violations = [
        f"{name}={actual!r} (required {required!r})"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if violations:
        raise ValueError(
            "base-auxiliary-pseudo invariant violation: " + "; ".join(violations)
        )


def resolve_optimizer_steps(
    *, schedule_mode: str, configured_steps: int, num_epochs: int,
    loader_microbatches: int, gradient_accumulation_steps: int,
    real_items: int, pseudo_items: int,
) -> int:
    """Resolve the final optimizer-step count after dataset construction."""
    if schedule_mode not in SCHEDULE_MODES:
        raise ValueError(f"Unsupported schedule_mode={schedule_mode!r}")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if schedule_mode == "epochs":
        if num_epochs <= 0:
            raise ValueError("epochs schedule requires num_epochs > 0")
        microbatches = loader_microbatches * num_epochs
        if microbatches % gradient_accumulation_steps:
            raise ValueError(
                "Exact epochs require epoch microbatches*num_epochs to be "
                "divisible by gradient accumulation"
            )
        return microbatches // gradient_accumulation_steps
    if num_epochs != 0:
        raise ValueError(f"{schedule_mode} schedule requires num_epochs == 0")
    if configured_steps <= 0:
        raise ValueError("configured_steps must be positive")
    if schedule_mode == "base-auxiliary-pseudo":
        return configured_steps
    return configured_steps


def expected_sample_exposure(
    *, real_items: int, pseudo_items: int, optimizer_steps: int,
    global_batch_size: int, base_reference_steps: int,
    schedule_mode: str = "steps",
    real_draw_fraction: float | None = None,
) -> dict[str, float | int | str]:
    """Describe expected source draws for plain uniform union sampling."""
    if real_items <= 0 or pseudo_items < 0:
        raise ValueError("real_items must be positive and pseudo_items non-negative")
    if optimizer_steps <= 0 or global_batch_size <= 0:
        raise ValueError("optimizer_steps and global_batch_size must be positive")
    union_items = real_items + pseudo_items
    total_draws = optimizer_steps * global_batch_size
    real_ratio = (
        real_items / union_items
        if real_draw_fraction is None
        else real_draw_fraction
    )
    if not 0.0 < real_ratio <= 1.0:
        raise ValueError("real_draw_fraction must be in (0, 1]")
    pseudo_ratio = pseudo_items / union_items
    if real_draw_fraction is not None:
        pseudo_ratio = 1.0 - real_ratio
    expected_real_draws = total_draws * real_ratio
    expected_pseudo_draws = total_draws * pseudo_ratio
    report: dict[str, float | int | str] = {
        "optimizer_step_formula": "optimizer steps resolved by the selected schedule mode",
        "expected_total_sample_draws": total_draws,
        "expected_real_sample_draws": expected_real_draws,
        "expected_pseudo_sample_draws": expected_pseudo_draws,
        "expected_pseudo_to_real_draw_ratio": expected_pseudo_draws / expected_real_draws,
        "expected_real_exposures_per_item": expected_real_draws / real_items,
        "expected_pseudo_exposures_per_item": (
            expected_pseudo_draws / pseudo_items if pseudo_items else 0.0
        ),
        "exposure_guarantee": (
            "No Base real-stream guarantee is claimed for this legacy mixed mode."
        ),
    }
    return report


def auxiliary_schedule_report(
    *,
    optimizer_steps: int,
    world_size: int,
    real_global_batch: int = 64,
    pseudo_global_batch: int = 8,
    pseudo_loss_weight: float = 0.25,
    pseudo_loss_warmup_steps: int = 3000,
) -> dict[str, int | float | str]:
    if optimizer_steps <= 0 or world_size <= 0:
        raise ValueError("optimizer_steps and world_size must be positive")
    if real_global_batch != 64 or real_global_batch % world_size:
        raise ValueError("real global batch 64 must be divisible by world_size")
    if pseudo_global_batch <= 0 or pseudo_global_batch % world_size:
        raise ValueError(
            "pseudo_global_batch must be positive and divisible by world_size"
        )
    if pseudo_loss_weight < 0:
        raise ValueError("pseudo_loss_weight must be non-negative")
    if pseudo_loss_warmup_steps < 0:
        raise ValueError("pseudo_loss_warmup_steps must be non-negative")
    real_accumulation = real_global_batch // world_size
    pseudo_microbatches = pseudo_global_batch // world_size
    update_plan = auxiliary_update_plan(
        real_microbatches=real_accumulation,
        pseudo_microbatches=pseudo_microbatches,
        pseudo_loss_weight=pseudo_loss_weight,
    )
    pseudo_enabled = bool(update_plan["pseudo_enabled"])
    return {
        "optimizer_steps": optimizer_steps,
        "real_global_batch": real_global_batch,
        "pseudo_global_batch": pseudo_global_batch,
        "real_microbatches_per_rank_update": real_accumulation,
        "pseudo_microbatches_per_rank_update": pseudo_microbatches,
        "pseudo_loss_weight": pseudo_loss_weight,
        "pseudo_loss_warmup_steps": pseudo_loss_warmup_steps,
        "pseudo_backward_scale": (
            pseudo_loss_weight * real_accumulation / pseudo_microbatches
        ),
        "expected_real_sample_draws": optimizer_steps * real_global_batch,
        "expected_pseudo_sample_draws": (
            optimizer_steps * pseudo_global_batch if pseudo_enabled else 0
        ),
        "objective": "mean(real_latent+real_action)+weight*mean(pseudo_latent+pseudo_action)",
        "pseudo_data_draws": "enabled" if pseudo_enabled else "disabled",
        "fsdp_sync": (
            "final_real_backward_then_independent_final_pseudo_backward"
            if pseudo_enabled
            else "final_real_backward_only"
        ),
    }


def should_rebuild_lr_scheduler(
    *, schedule_mode: str, scheduler_type: str,
    trainer_steps: int, final_steps: int,
) -> bool:
    """Whether dataset-dependent steps invalidate the Trainer's scheduler."""
    normalized = scheduler_type.strip().lower()
    return (
        schedule_mode == "epochs"
        and final_steps != trainer_steps
        and normalized in TOTAL_STEP_DEPENDENT_SCHEDULERS
    )
