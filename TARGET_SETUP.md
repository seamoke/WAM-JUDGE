# H100 Target Setup Notes

> Historical record for the original four-H100 machine. New deployments should
> follow `auto_pipline_readme.md`; its paths and portable entry points are
> authoritative.

Target hardware observed on 2026-07-22:

- Ubuntu 22.04
- 4 x NVIDIA H100 80GB
- NVIDIA driver 570.172.08
- Driver CUDA compatibility 12.8
- `/opt/conda` exists
- CUDA toolkit (`nvcc`) and GCC are not initially installed

## Environment layout

- Code: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code`
- Conda env: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/env`
- Models: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/models`
- Datasets: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/datasets`
- RoboTwin assets: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/robotwin-assets`
- Runs and logs: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/runs`

## Required downloads not present in the code archive

- Base training model: `Robbyant/lingbot-va-base`
- Official evaluation model: `Robbyant/lingbot-va-posttrain-robotwin`
- Training data: `Robbyant/robotwin-clean-and-aug-lerobot`
- RoboTwin background, embodiment, and object assets

## Protocol invariants

- RoboTwin commit `2eeec322d95799f537cbfe5f291a8220d965ccb8`
- Official LingBot-VA action chunk: 2
- Video diffusion steps: 25
- Action diffusion steps: 50
- Video guidance: 5
- Action guidance: 1
- Evaluation uses RT rendering unless an experiment explicitly says otherwise
- Easy configuration: `demo_clean`
- Hard configuration: `demo_randomized`

The destination setup must compile FlashAttention, PyTorch3D, and Curobo against the
destination PyTorch/CUDA environment. Compiled `.so` files from the source Pod are not
portable and are intentionally excluded.
