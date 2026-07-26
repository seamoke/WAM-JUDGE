"""Launch ms-swift SFT with a Transformers 4.51 DDP loading compatibility fix."""

from __future__ import annotations

from functools import wraps
from typing import Callable


def compatible_caching_allocator_warmup(original: Callable) -> Callable:
    """Avoid treating a missing tensor-parallel plan as an iterable in DDP."""

    @wraps(original)
    def wrapped(model, *args, **kwargs):
        original_tp_plan = getattr(model, "_tp_plan", None)
        needs_compat = original_tp_plan is None
        if needs_compat:
            # A non-matching sentinel keeps DDP allocations unsharded while
            # satisfying Transformers 4.51's unconditional regex construction.
            model._tp_plan = ["__vlac_ddp_no_tensor_parallel_plan__"]
        try:
            return original(model, *args, **kwargs)
        finally:
            if needs_compat:
                model._tp_plan = original_tp_plan

    wrapped._vlac_ddp_compat = True
    return wrapped


def compatible_internvl_loss_context(original: Callable) -> Callable:
    """Give the InternVL template the underlying model when training with DDP."""

    @wraps(original)
    def wrapped(template, model, inputs):
        return original(template, getattr(model, "module", model), inputs)

    wrapped._vlac_ddp_compat = True
    return wrapped


def install_compatibility_patch() -> None:
    import transformers.modeling_utils as modeling_utils
    from swift.llm.template.template.internvl import InternvlTemplate

    current = modeling_utils.caching_allocator_warmup
    if not getattr(current, "_vlac_ddp_compat", False):
        modeling_utils.caching_allocator_warmup = compatible_caching_allocator_warmup(
            current
        )

    loss_context = InternvlTemplate.compute_loss_context
    if not getattr(loss_context, "_vlac_ddp_compat", False):
        InternvlTemplate.compute_loss_context = compatible_internvl_loss_context(
            loss_context
        )


def main() -> None:
    install_compatibility_patch()
    from swift.llm import sft_main

    sft_main()


if __name__ == "__main__":
    main()
