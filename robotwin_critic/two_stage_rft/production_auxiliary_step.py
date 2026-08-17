"""Shared production auxiliary RFT optimizer step.

This module is intentionally importable by the equivalence audit: the audit and
normal joint RFT must execute this exact function, not parallel implementations.
"""

from __future__ import annotations

import hashlib
import inspect

import torch
import torch.distributed as dist

from robotwin_critic.two_stage_rft.rft_schedule import effective_pseudo_loss_weight
from robotwin_critic.two_stage_rft.rft_schedule import auxiliary_scaling_config


# The official precomputed-text path bypasses this internal text projection.
# These trainable parameters are also gradient-free in Base SFT; every other
# trainable parameter remains subject to the strict missing-gradient gate.
STRUCTURALLY_UNUSED_GRADIENTS = frozenset(
    {
        "condition_embedder_action.text_embedder.linear_1.weight",
        "condition_embedder_action.text_embedder.linear_1.bias",
        "condition_embedder_action.text_embedder.linear_2.weight",
        "condition_embedder_action.text_embedder.linear_2.bias",
    }
)


def implementation_provenance() -> dict[str, str]:
    """Fingerprint every shared implementation the audit must not duplicate."""
    step_source = inspect.getsource(production_auxiliary_train_step).encode("utf-8")
    initializer_source = inspect.getsource(
        initialize_production_auxiliary_state
    ).encode("utf-8")
    scaling_source = inspect.getsource(auxiliary_scaling_config).encode("utf-8")
    return {
        "module": production_auxiliary_train_step.__module__,
        "qualname": production_auxiliary_train_step.__qualname__,
        "source_sha256": hashlib.sha256(step_source).hexdigest(),
        "initializer_module": initialize_production_auxiliary_state.__module__,
        "initializer_qualname": initialize_production_auxiliary_state.__qualname__,
        "initializer_source_sha256": hashlib.sha256(initializer_source).hexdigest(),
        "scaling_module": auxiliary_scaling_config.__module__,
        "scaling_qualname": auxiliary_scaling_config.__qualname__,
        "scaling_source_sha256": hashlib.sha256(scaling_source).hexdigest(),
    }


def initialize_production_auxiliary_state(
    trainer, *, train_module, pseudo_loss_weight: float,
    pseudo_loss_warmup_steps: int, pseudo_sampler_seed: int,
    pseudo_microbatches_per_update: int,
) -> None:
    """Initialize the state consumed by the actual production auxiliary step."""
    trainer.train_module = train_module
    trainer.pseudo_loss_weight = pseudo_loss_weight
    trainer.pseudo_loss_warmup_steps = pseudo_loss_warmup_steps
    trainer.pseudo_sampler_seed = pseudo_sampler_seed
    trainer.pseudo_microbatches_per_update = pseudo_microbatches_per_update
    scaling = auxiliary_scaling_config(
        real_microbatches=int(trainer.gradient_accumulation_steps),
        pseudo_microbatches=pseudo_microbatches_per_update,
        pseudo_loss_weight=pseudo_loss_weight,
    )
    trainer.auxiliary_update_plan = scaling["auxiliary_update_plan"]
    trainer.pseudo_enabled = bool(trainer.auxiliary_update_plan["pseudo_enabled"])
    trainer.pseudo_backward_scale_per_weight = scaling[
        "pseudo_backward_scale_per_weight"
    ]
    trainer._aux_real_latent = torch.zeros((), device=trainer.device)
    trainer._aux_real_action = torch.zeros((), device=trainer.device)


def _local_value(value):
    """Return the rank-local tensor for a DTensor, or the tensor itself."""
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _gradient_is_finite(gradient) -> bool:
    """Check only the local shard so scalar conversion never targets a DTensor."""
    return bool(torch.isfinite(_local_value(gradient)).all())


def _collective_gradient_validation(
    trainer,
    boundary: str,
    *,
    incompatible=(),
) -> list[tuple[str, object, object]]:
    """Check all local gradients, then reach one agreement before raising."""
    gradients = []
    missing = []
    nonfinite = []
    trainable_count = 0
    for name, parameter in trainer.transformer.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_count += 1
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        if not _gradient_is_finite(gradient):
            nonfinite.append(name)
        gradients.append((name, parameter, gradient))

    unexpected_missing = [
        name for name in missing if name not in STRUCTURALLY_UNUSED_GRADIENTS
    ]
    empty = trainable_count == 0 or not gradients
    local_success = not (
        unexpected_missing or nonfinite or empty or incompatible
    )
    success = torch.tensor(local_success, dtype=torch.int32, device=trainer.device)
    if dist.is_available() and dist.is_initialized():
        # No process group is supplied: FSDP2 uses the default data-parallel world.
        dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if bool(success.item()):
        return gradients

    local_details = []
    if unexpected_missing:
        local_details.append(f"local missing={list(unexpected_missing)}")
    if nonfinite:
        local_details.append(f"local nonfinite={list(nonfinite)}")
    if empty:
        local_details.append("local trainable gradient set is empty")
    if incompatible:
        local_details.append(f"local incompatible={list(incompatible)}")
    detail = f" ({'; '.join(local_details)})" if local_details else ""
    message = f"{boundary}: gradient validation failed on one or more ranks{detail}"
    if nonfinite and not (unexpected_missing or empty or incompatible):
        raise FloatingPointError(message)
    raise RuntimeError(message)


def _validated_gradient_shards(trainer, boundary: str):
    """Clone reduced shards after one collective missing/finite validation."""
    gradients = _collective_gradient_validation(trainer, boundary)
    return [
        (name, parameter, gradient.detach().clone())
        for name, parameter, gradient in gradients
    ]


def _matching_public_attribute(values, attribute: str) -> bool:
    metadata = [getattr(value, attribute, None) for value in values]
    if not any(item is not None for item in metadata):
        return True
    if any(item is None for item in metadata):
        return False
    first = metadata[0]
    try:
        return all(bool(item == first) for item in metadata[1:])
    except (RuntimeError, TypeError, ValueError):
        return all(item is first for item in metadata[1:])


def _compatible_gradient_shards(parameter, current, real, pseudo) -> bool:
    """Validate public DTensor metadata before DTensor-native in-place merging."""
    values = (parameter, current, real, pseudo)
    if any(value.shape != parameter.shape for value in values[1:]):
        return False
    if not _matching_public_attribute(values, "placements"):
        return False
    if not _matching_public_attribute(values, "device_mesh"):
        return False
    local_shapes = [tuple(_local_value(value).shape) for value in values]
    return all(shape == local_shapes[0] for shape in local_shapes[1:])


def production_auxiliary_train_step(trainer, batch, batch_idx):
    """Run the production real-plus-optional-pseudo auxiliary update."""
    accumulation = int(trainer.gradient_accumulation_steps)
    trainer.transformer.set_requires_gradient_sync(
        trainer.auxiliary_update_plan["real_sync_flags"][batch_idx]
    )
    batch_size = int(batch["latents"].shape[0])
    batch = trainer.convert_input_format(batch)
    input_dict = trainer._prepare_input_dict(batch)
    output = trainer.transformer(input_dict, train_mode=True)
    real_latent, real_action = trainer.compute_loss(input_dict, output)
    (real_latent + real_action).backward()
    trainer._aux_real_latent += real_latent.detach()
    trainer._aux_real_action += real_action.detach()
    trainer.rft_source_counts[0] += batch_size
    losses = {"latent_loss": real_latent.detach(), "action_loss": real_action.detach(), "should_log": False}
    if batch_idx + 1 != accumulation:
        return losses

    # compute_loss already divides each loss by A=gradient_accumulation_steps.
    # The synchronized real shards therefore equal (1/A) sum_i grad(L_real_i),
    # exactly Base.  Preserve them while pseudo receives its own FSDP2 reduction.
    real_gradient_shards = None
    if trainer.pseudo_enabled:
        real_gradient_shards = _validated_gradient_shards(trainer, "real boundary")
        observer = getattr(trainer, "_observe_real_boundary", None)
        if observer is not None:
            observer({name: shard.clone() for name, _, shard in real_gradient_shards})
        trainer.optimizer.zero_grad()

    pseudo_latent_mean = torch.zeros((), device=trainer.device)
    pseudo_action_mean = torch.zeros((), device=trainer.device)
    effective_pseudo_weight = effective_pseudo_loss_weight(
        target_weight=trainer.pseudo_loss_weight,
        warmup_steps=trainer.pseudo_loss_warmup_steps,
        optimizer_step=int(trainer.step),
    )
    for pseudo_index, sync_pseudo in enumerate(trainer.auxiliary_update_plan["pseudo_sync_flags"]):
        seed = trainer.pseudo_sampler_seed * 1_000_003 + int(trainer.step) * 10_007 + int(trainer.config.rank) * 101 + pseudo_index
        devices = [trainer.device.index] if trainer.device.type == "cuda" and trainer.device.index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if trainer.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            pseudo_batch = trainer._get_next_pseudo_batch()
            pseudo_batch_size = int(pseudo_batch["latents"].shape[0])
            pseudo_batch = trainer.convert_input_format(pseudo_batch)
            pseudo_input = trainer._prepare_input_dict(pseudo_batch)
            trainer.transformer.set_requires_gradient_sync(sync_pseudo)
            pseudo_output = trainer.transformer(pseudo_input, train_mode=True)
            pseudo_latent, pseudo_action = trainer.compute_loss(pseudo_input, pseudo_output)
            ((pseudo_latent + pseudo_action) * effective_pseudo_weight * trainer.pseudo_backward_scale_per_weight).backward()
        pseudo_latent_mean += pseudo_latent.detach() * accumulation / trainer.pseudo_microbatches_per_update
        pseudo_action_mean += pseudo_action.detach() * accumulation / trainer.pseudo_microbatches_per_update
        trainer.rft_source_counts[1] += pseudo_batch_size

    if real_gradient_shards is not None:
        pseudo_gradient_shards = _validated_gradient_shards(trainer, "pseudo boundary")
        pseudo_by_name = {name: (parameter, shard) for name, parameter, shard in pseudo_gradient_shards}
        real_names = {name for name, _, _ in real_gradient_shards}
        incompatible = [
            name for name in pseudo_by_name if name not in real_names
        ]
        for name, parameter, real_shard in real_gradient_shards:
            pseudo_entry = pseudo_by_name.get(name)
            if pseudo_entry is None or parameter.grad is None:
                incompatible.append(name)
                continue
            pseudo_parameter, pseudo_shard = pseudo_entry
            if pseudo_parameter is not parameter or not _compatible_gradient_shards(
                parameter, parameter.grad, real_shard, pseudo_shard
            ):
                incompatible.append(name)
                continue
            try:
                # Preserve DTensor dispatch; the shards were detached and cloned above.
                parameter.grad.copy_(real_shard).add_(pseudo_shard)
            except (RuntimeError, TypeError, ValueError):
                incompatible.append(name)
                continue
        _collective_gradient_validation(
            trainer,
            "combined boundary",
            incompatible=incompatible,
        )

    total_norm = torch.nn.utils.clip_grad_norm_(trainer.transformer.parameters(), 2.0)
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad()
    # dist_mean may return the accumulator itself. Clone the metric snapshot so
    # resetting the accumulator below cannot also zero the value sent to logs.
    real_latent_mean = trainer.train_module.dist_mean(trainer._aux_real_latent).detach().clone()
    real_action_mean = trainer.train_module.dist_mean(trainer._aux_real_action).detach().clone()
    pseudo_latent_mean = trainer.train_module.dist_mean(pseudo_latent_mean).detach()
    pseudo_action_mean = trainer.train_module.dist_mean(pseudo_action_mean).detach()
    combined_update_loss = real_latent_mean + real_action_mean + effective_pseudo_weight * (pseudo_latent_mean + pseudo_action_mean)
    trainer.last_aux_loss_metrics = {
        "last_real_latent_loss_unscaled": real_latent_mean.item(),
        "last_real_action_loss_unscaled": real_action_mean.item(),
        "last_pseudo_latent_loss_unscaled": pseudo_latent_mean.item(),
        "last_pseudo_action_loss_unscaled": pseudo_action_mean.item(),
        "last_combined_update_loss": combined_update_loss.item(),
        "pseudo_loss_weight_effective": effective_pseudo_weight,
    }
    trainer._aux_real_latent.zero_()
    trainer._aux_real_action.zero_()
    losses.update({
        "should_log": True, "total_norm": total_norm,
        "real_latent_loss_unscaled": real_latent_mean, "real_action_loss_unscaled": real_action_mean,
        "pseudo_latent_loss_unscaled": pseudo_latent_mean, "pseudo_action_loss_unscaled": pseudo_action_mean,
        "combined_update_loss": combined_update_loss, "pseudo_loss_weight_effective": effective_pseudo_weight,
    })
    return losses
