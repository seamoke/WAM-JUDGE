# Upstream VLAC runtime files

Source: https://github.com/InternRobotics/VLAC

Retrieved from the `main` branch on 2026-07-15. Only the small Python runtime
files needed by `evo_vlac.GAC_model` and the upstream `pyproject.toml` are
mirrored here. Model weights are not vendored.

The official `model_utils.py` uses top-level imports for `data_processing_vlm`
and `video_tool`. The sidecar evaluator adds both this directory and
`evo_vlac/utils` to `sys.path` instead of modifying upstream source files.
