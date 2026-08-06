# RoboTwin Stage-2 Dual-RFT

This pipeline calibrates a kinematic Action Critic, builds action-free Stage-2
contexts, generates WAM chunk candidates, filters them with Action Critic and
VLAC, and repeatedly fine-tunes the full WAM transformer.

The canonical entry point is:

```text
script/run_robotwin_stage2_online_rft_pipeline.sh
```

## Inputs

The command requires three paths:

1. `--stage2-data-root`: either the prepared split root or its `stage2/`
   directory.
2. `--wam-model`: a complete WAM root or a transformer-only checkpoint such as
   `checkpoint_step_15000`.
3. `--vlac-model`: the trained VLAC checkpoint used as Process Critic.

The prepared data must have this layout:

```text
prepared_dataset/
├── split_manifest.json
├── PREPARATION_COMPLETE.json
├── stage1/                  # action-labeled trajectories
└── stage2/                  # action-free trajectories
```

Although the CLI accepts the `stage2/` path, `stage1/` must be its sibling.
The Action Critic is calibrated only from Stage-1 actions. Stage-2 actions are
not read.

The Kinematic Action Critic is a calibrated profile rather than a neural
network checkpoint. It stores task-aware velocity, acceleration, jerk,
displacement, gripper, rotation, and workspace thresholds in JSON. The current
50-task profile is 353KB: it was calibrated from 3,000 Stage-1 trajectories and
40,094 action chunks, and requires no GPU memory at inference time.

## One-command run

```bash
cd /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code

bash script/run_robotwin_stage2_online_rft_pipeline.sh \
  --stage2-data-root \
    /inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/datasets/robotwin-clean-aug-two-stage-redacted-seed42/stage2 \
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

1. Calibrate the Action Critic from Stage-1 action-labeled trajectories.
2. Build Stage-2 video contexts without reading Stage-2 action labels.
3. Build the Stage-2 pseudo-chunk budget.
4. Compose a complete WAM root when the input checkpoint contains only a
   transformer.
5. For each collection round, sample 16 contexts and generate 8 one-chunk
   candidates per context.
6. Reject candidates below the Action Critic threshold.
7. Score the remaining candidate futures with VLAC and reject low process
   scores.
8. Append every accepted candidate from the same context to a 512-chunk replay
   buffer.
9. When the buffer is full, train for 3 epochs using 50% Stage-1 real chunks
   and 50% pseudo chunks with global batch 64.
10. Replace the WAM transformer with the updated checkpoint and repeat.

Only the WAM transformer is trainable, but it is fine-tuned in full rather than
with LoRA. Both video latent flow-matching loss and action flow-matching loss
are optimized.

## Default hyperparameters

| Setting | Default |
|---|---:|
| GPUs | `0,1,2,3` |
| contexts per collection | 16 |
| candidates per context | 8 |
| pseudo-buffer capacity | 512 |
| real/pseudo sampling | 50% / 50% |
| pseudo epochs per update | 3 |
| global training batch | 64 |
| Action Critic threshold | 0.75 |
| VLAC process threshold | 5.0 |
| maximum RFT updates | 1000 |

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
