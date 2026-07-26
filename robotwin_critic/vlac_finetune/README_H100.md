# RoboTwin VLAC-2B Fine-Tuning on 4x H100

This module is standalone. It does not modify or import the existing WAM
training, evaluation, server, or client entry points.

## Server Paths

- Project: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code`
- Dataset: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/datasets/robotwin-clean-and-aug-lerobot`
- Outputs: `/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/train_out/critic/robotwin`

## 1. Download and Build Data

Run:

```bash
cd /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code
BUILD_FULL=0 script/robotwin_vlac_prepare_h100.sh
```

The script:

1. Creates an isolated, system-site-packages venv.
2. Installs the pinned VLAC/ms-swift dependencies.
3. Downloads `InternRobotics/VLAC` directly on the server. It tries the
   official Hugging Face endpoint first and `hf-mirror.com` second.
4. Validates that `config.json` and a safetensors file larger than 1 GiB
   exist.
5. Builds the RoboTwin RGB index.
6. Builds and validates a two-task smoke dataset.

Set `BUILD_FULL=1` to build all RGB pair data in the same command. Full data
generation defaults to eight CPU workers and can be controlled with
`DATA_WORKERS`.

Important output paths:

- Model: `train_out/critic/robotwin/models/VLAC-2B`
- RGB index: `train_out/critic/robotwin/index_rgb.jsonl`
- Smoke data: `train_out/critic/robotwin/vlac_finetune/smoke_2task`
- Full data: `train_out/critic/robotwin/vlac_finetune/full`

## 2. Smoke Train and Evaluate

Run only after all four GPUs are idle:

```bash
cd /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code
script/robotwin_vlac_train_4xh100.sh smoke
```

The training script requires exactly four visible GPUs and refuses to start
when `nvidia-smi` reports any existing compute process. The smoke workflow:

1. Evaluates the original VLAC-2B checkpoint on the smoke validation set.
2. Runs ten 4-GPU LoRA optimization steps.
3. Evaluates the LoRA adapter on the identical validation subset.
4. Verifies RGB loading, numeric score parsing, VOC, and VROC.
5. Writes a baseline-versus-LoRA metric comparison.

Smoke outputs:

- Adapter: `vlac_finetune/vlac_2b_lora_smoke_4xh100`
- Baseline metrics: `vlac_finetune/vlac_2b_eval_baseline_4xh100/summary.json`
- LoRA metrics: `vlac_finetune/vlac_2b_eval_smoke_4xh100/summary.json`
- Comparison: `vlac_finetune/vlac_2b_eval_smoke_4xh100/comparison_vs_baseline.json`

## 3. Full Train

After smoke gates and full data validation pass:

```bash
cd /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code
script/robotwin_vlac_train_4xh100.sh full
```

By default, full training runs one complete epoch. Set `MAX_STEPS` only for
an explicit bounded experiment:

```bash
MAX_STEPS=1000 script/robotwin_vlac_train_4xh100.sh full
```

The effective global batch size is 16:

- 4 GPUs
- batch size 1 per GPU
- gradient accumulation 4

Full checkpoints and TensorBoard logs are written under:

`train_out/critic/robotwin/vlac_finetune/vlac_2b_lora_full_4xh100`

## Data and Metrics

Each RGB state is a 448x448 T-shaped mosaic of the high, left-wrist, and
right-wrist cameras. Train/validation splitting is episode-level.

For every sampled forward pair, the dataset stores the exact reversed pair
with the negated target. Static adjacent pairs form the neutral class.
Corrupt episodes are skipped transactionally, so a partially decoded episode
never contributes samples.

Evaluation reports:

- Numeric parse rate
- Pair MAE and sign accuracy
- Negative/neutral/positive macro-F1
- Three-class one-vs-rest AUC
- Pair target Spearman correlation
- VOC and VROC
- VOC/VROC harmonic mean
- Forward/reverse antisymmetry MAE

