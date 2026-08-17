# RoboTwin Stage-2 Dual-RFT

This pipeline calibrates a kinematic Action Critic, builds action-free Stage-2
contexts, generates WAM chunk candidates, filters them with Action Critic and
VLAC, and repeatedly fine-tunes the full WAM transformer.

## Current one-shot RFT training

The current production path can also skip online collection and train directly
from the saved 20,477-row pseudo buffer. Experiment A initializes from the
Stage-1 SFT `checkpoint_step_15000`. Experiment B intentionally initializes
from the original `lingbot-va-base` model to reproduce Stage-1 training through
the current RFT code path. The main experiment optimizes:

```text
mean(Stage-1 + Stage-2 real latent_loss + action_loss)
  + lambda * mean(pseudo latent_loss + action_loss)
```

The real stream uses the official full-chunk loader and official Base SFT
`latent_loss + action_loss`. The pseudo stream is an independent auxiliary
gradient, with `lambda` linearly warmed up during the first 100 optimizer
updates. Both streams draw 64 samples per optimizer update by default.

The updated scheduler accepts any positive optimizer-step count. Checkpoint
steps are read from the configured save interval and are no longer replaced by
the old fixed `3000,6000,...,15000` schedule. Completion validation likewise
accepts the configured positive step count instead of requiring exactly 15,000.

### Experiment A: 8x MI355X real + pseudo

For one node with eight 280GB AMD MI355X GPUs, use:

```bash
cd /workspace/lingbot-training

bash script/run_stage1_stage2_real_pseudo_lambda01_8xmi355_16k.sh
```

The preset has the following effective settings:

| Setting | Value |
|---|---:|
| initializer | Stage-1 SFT `checkpoint_step_15000` |
| real data | all Stage-1 + Stage-2 real chunks |
| pseudo data | validated 20,477-row buffer |
| GPUs | `0,1,2,3,4,5,6,7` |
| optimizer steps | 16,000 |
| checkpoints | step 4,000, 8,000, 12,000 and 16,000 |
| real batch per GPU | 1 |
| real global batch | 64 |
| gradient accumulation | 8 |
| pseudo global batch | 64 |
| pseudo loss coefficient | 0.1 |
| pseudo warmup | 100 steps |
| activation checkpointing | off |
| trainable parameters | full transformer |

The real execution packing is `8 GPUs x 1 sample x GA 8 = 64`, matching the
Base SFT microbatch and accumulation boundary. Pseudo batches remain global batch 64.
The generic launcher validates that `batch_per_gpu x world_size` divides 64, so
invalid combinations fail before model loading.

Paths can be overridden without editing the script:

```bash
PROJECT_ROOT=/path/to/WAM-JUDGE \
LINGBOT_ROOT=/path/to/lingbot-va \
PREPARED_DATA_ROOT=/path/to/prepared-stage1-stage2 \
STAGE1_CHECKPOINT=/path/to/checkpoint_step_15000 \
PSEUDO_JSONL=/path/to/one_shot_pseudo_buffer.validated.jsonl \
RUN_ID=my-mi355-rft \
bash script/run_stage1_stage2_real_pseudo_lambda01_8xmi355_16k.sh
```

The output is written to
`$LINGBOT_ROOT/train_out/robotwin/$RUN_ID`. SwanLab metrics are uploaded to
`lingbot-va-robotwin` unless `LINGBOT_SWANLAB_PROJECT` is overridden.

### Experiment B: Stage-1 real-only loss control

Run a second, independent experiment to verify that the real-data loader and
official Base SFT loss do not themselves cause a regression:

```bash
cd /workspace/lingbot-training

bash script/run_stage1_real_from_base_8xmi355_15k.sh
```

This control initializes from the original `lingbot-va-base` model and trains
for the full 15,000-step Stage-1 schedule using only the complete Stage-1 real
dataset. It uses the same `1e-5` learning rate, constant scheduler, 10-step
warmup and global batch 64 as the original Stage-1 SFT. To reduce storage and
transfer, it saves only the final step-15,000 model. Its pseudo coefficient is
exactly zero, so no pseudo sample contributes a forward pass, gradient, or
source-count update.

This is enforced as a runtime invariant at every optimizer-update boundary:
when `PSEUDO_LOSS_WEIGHT=0`, the pseudo source count must remain zero. The
pseudo JSONL is still read during CPU-side startup validation, but no pseudo
batch is fetched for model execution and no pseudo forward/backward is run.

| Setting | Value |
|---|---:|
| initializer | original `lingbot-va-base` |
| real data | all Stage-1 real chunks only |
| pseudo loss coefficient | 0 |
| GPUs | `0,1,2,3,4,5,6,7` |
| optimizer steps | 15,000 |
| checkpoints | final step 15,000 only |
| real batch per GPU | 1 |
| real global batch | 64 |
| gradient accumulation | 8 |
| activation checkpointing | off |
| trainable parameters | full transformer |

Evaluate Experiment B against the official Stage-1 15k checkpoint using the
same task list, seeds, evaluator, and inference settings. If Experiment B drops
substantially, investigate the real loader/loss or optimizer path before
attributing Experiment A's behavior to pseudo supervision. Experiment A and B
write separate timestamped output directories and should be run independently,
not resumed from one another.

The canonical entry point is:

```text
script/run_robotwin_stage2_online_rft_pipeline.sh
```

## Inputs

The command requires these paths:

1. `--stage2-data-root`: either the prepared split root or its `stage2/`
   directory.
2. `--wam-model`: a complete WAM root or a transformer-only checkpoint such as
   `checkpoint_step_15000`.
3. `--vlac-model`: the trained VLAC checkpoint used as Process Critic.
4. `--original-robotwin-root`: the original action-visible RoboTwin dataset.
   It is needed once to restore only the manifest-selected Stage-2 actions. It
   may be omitted when `split_manifest.json` still records a valid source path,
   or when `action_visible_real/` has already been built.

The prepared data must have this layout:

```text
prepared_dataset/
├── split_manifest.json
├── PREPARATION_COMPLETE.json
├── stage1/                  # action-labeled trajectories
├── stage2/                  # action-free trajectories
└── action_visible_real/     # generated selected 30+20 real replay view
```

Although the CLI accepts the `stage2/` path, `stage1/` must be its sibling.
The Action Critic is calibrated only from Stage-1 actions. Stage-2 actions are
not read while building contexts, calibrating critics, or selecting pseudo
chunks. They are restored only for the 50% real replay stream during WAM
fine-tuning.

To build that replay view before starting RFT:

```bash
bash script/prepare_robotwin_action_visible_real.sh \
  --prepared-root /path/to/prepared_dataset \
  --source-root /path/to/original_robotwin \
  --output-root /path/to/prepared_dataset/action_visible_real \
  --link-mode hardlink
```

Use `--link-mode copy` when the resulting directory will itself be transferred
to another machine. The main pipeline runs this step automatically when the
completion marker is absent.

The Kinematic Action Critic is a calibrated profile rather than a neural
network checkpoint. It stores task-aware velocity, acceleration, jerk,
displacement, gripper, rotation, and workspace thresholds in JSON. The current
50-task profile is 353KB: it was calibrated from 3,000 Stage-1 trajectories and
40,094 action chunks, and requires no GPU memory at inference time.

## One-command run

For a reusable no-argument command, copy
`script/robotwin_stage2_online_rft.env.example` to
`code/.local/robotwin_stage2_rft.env`, fill in the local paths, and run:

```bash
cd /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code
./script/run_robotwin_stage2_online_rft_pipeline.sh
```

The `.local` configuration is host-specific and should not be committed. CLI
arguments remain available and take precedence over values in the local file.

The equivalent explicit command is:

```bash
cd /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code

bash script/run_robotwin_stage2_online_rft_pipeline.sh \
  --stage2-data-root \
    /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/datasets/robotwin-clean-aug-two-stage-redacted-seed42/stage2 \
  --original-robotwin-root \
    /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/datasets/robotwin-clean-and-aug-lerobot \
  --wam-model \
    /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/models/lingbot-va-stage1-checkpoints/checkpoint_step_15000 \
  --vlac-model \
    /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/train_out/critic/robotwin/vlac_finetune/vlac_2b_full_4xh100/v0-20260724-175122/checkpoint-43852 \
  --output-root \
    /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/train_out/critic/robotwin/my_stage2_rft
```

For a transformer-only WAM checkpoint, the script takes VAE, tokenizer, and
text encoder from:

```text
/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/models/lingbot-va-base
```

Override it with `--base-model PATH` when necessary.

To run in the background:

```bash
nohup bash script/run_robotwin_stage2_online_rft_pipeline.sh \
  --stage2-data-root /path/to/prepared_dataset/stage2 \
  --wam-model /path/to/checkpoint_step_15000 \
  --vlac-model /path/to/vlac_checkpoint \
  --output-root /path/to/output \
  > /path/to/launcher.log 2>&1 &
```

The SwanLab API key is read from `SWANLAB_API_KEY` or
`code/.secrets/swanlab_api_key`.

## Execution order

The command performs these stages in order:

1. Reconstruct the exact manifest-selected 30+20 action-visible real replay
   view from the original dataset, then verify every parquet action column.
2. Calibrate the Action Critic from Stage-1 action-labeled trajectories only.
3. Build Stage-2 video contexts without reading Stage-2 action labels.
4. Build the Stage-2 pseudo-chunk budget.
5. Compose a complete WAM root when the input checkpoint contains only a
   transformer.
6. For each collection round, sample 128 contexts and generate 8 one-chunk
   candidates per context.
7. Reject candidates below the Action Critic threshold.
8. Score the remaining candidate futures with VLAC and reject low process
   scores.
9. Append every accepted candidate from the same context to a 1024-chunk replay
   buffer.
10. When the buffer is full, train for 3 epochs using 50% selected Stage-1 +
    Stage-2 real chunks and 50% pseudo chunks with global batch 32. Four GPUs
    each process batch 8, so gradient accumulation is 1.
11. Replace the WAM transformer with the updated checkpoint and repeat.

The rolling model is refreshed after every RFT update so the next collection
uses the latest parameters. Historical model checkpoints are retained only
every 50 updates (`50, 100, 150, ...`). Non-milestone rolling models are
removed after their successor is active.

Only the WAM transformer is trainable, but it is fine-tuned in full rather than
with LoRA. Both video latent flow-matching loss and action flow-matching loss
are optimized.

## Default hyperparameters

| Setting | Default |
|---|---:|
| GPUs | `0,1,2,3` |
| contexts per collection | 128 |
| candidates per context | 8 |
| pseudo-buffer capacity | 1024 |
| real/pseudo sampling | 50% / 50% |
| pseudo epochs per update | 3 |
| training batch per GPU | 8 |
| global training batch | 32 |
| gradient accumulation | 1 |
| Action Critic threshold | 0.75 |
| VLAC process threshold | 5.0 |
| maximum RFT updates | 1000 |
| retained model interval | 50 RFT updates |

Use `--help` to list the corresponding CLI overrides.

## Outputs

```text
OUTPUT_ROOT/
├── stage2_online_rft.log
├── part2/
│   ├── stage1_kinematic_profile.json
│   ├── stage2_video_contexts.jsonl
│   └── stage2_chunk_budget.json
└── online/
    ├── initial_model/
    ├── state.json
    ├── pending_buffer.jsonl
    ├── collect/collect_XXXXXX/
    │   ├── candidates.jsonl
    │   ├── action_scored.jsonl
    │   ├── dual_scored.jsonl
    │   ├── selected_winners.jsonl
    │   └── selection_summary.json
    ├── buffers/
    ├── updates/
    ├── checkpoints/rft_update_000050 -> ../updates/update_000049/model
    ├── swanlab_url.txt
    └── swanlab/
```

`selected_winners.jsonl` contains every candidate that passes both critics;
the name is retained for compatibility and does not imply one winner per
context.

## Prepare, resume, and restart

Build only the Action Critic and Stage-2 contexts:

```bash
bash script/run_robotwin_stage2_online_rft_pipeline.sh \
  --stage2-data-root /path/to/prepared_dataset/stage2 \
  --wam-model /path/to/wam \
  --vlac-model /path/to/vlac \
  --output-root /path/to/output \
  --prepare-only
```

To resume, rerun the same command with the same `--output-root`. The pipeline
reuses its profile, contexts, buffer, current model, and SwanLab run ID.

To discard the run and restart from the supplied WAM checkpoint, add
`--fresh`. This permanently deletes only `OUTPUT_ROOT`; it never deletes the
input dataset or model checkpoints.

## Monitoring

The long-lived Python RFT process owns one SwanLab session for the complete
run. Collection retention, Action/VLAC score distributions, task coverage,
buffer fill, training losses, gradient norm, learning rate, GPU memory, and
real/pseudo sampling ratios are written to the same run. SwanLab is finished
only when the RFT process exits.
