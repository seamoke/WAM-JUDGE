# Part 2: Chunk-Level Dual-RFT

Part 2 从 Part 1 的 M30 checkpoint 出发。原 WAM train/eval/server/client 文件不修改，
全部新增代码位于 `robotwin_critic/two_stage_rft/`。

## 方法

每个 task 固定使用 50 Clean + 50 Randomized：

| 数据 | Clean | Randomized | action 可见性 |
|---|---:|---:|---|
| Stage 1 / D30 | 30 | 30 | 可见 |
| Stage 2 / D20 | 20 | 20 | 隐藏 |

1. Kinematic Action Critic 只读取 D30 action，校准双臂 EEF velocity、
   acceleration、jerk、rotation、gripper 和可选 workspace 阈值。profile 保存 split
   哈希和全部校准 episode ID。
2. D20 每条视频在 10%、30%、50%、70%、90% 进度构造 context。代码只读取三路
   RGB、proprio 和 language，不读取 action。
3. M30 对每个 context 生成 8 个 `(video, action)` chunk。
4. Dual-RFT 先执行 `action_score > threshold`，再用 VLAC process score 选每个
   context 的最佳候选；没有 action 合格候选就丢弃该 context。
5. pseudo budget 等于 D20 在原 WAM loader 中的有效 chunk 数，并按 task/domain
   分组严格匹配。数量不足时直接失败，不降低阈值。
6. RFT batch 为 70% D30 real + 30% pseudo。只优化
   `action_embedder`、`condition_embedder_action`、`action_proj_out`，video/shared
   backbone 冻结，loss 只有 action flow matching。

当前官方 WAM 首 chunk 接口只接收当前 RGB 和 language。context 仍保存 2–4 帧
history 与 proprio，但生成时不会假称模型使用了不存在的 proprio/history token。
若要注入它们，应作为后续架构实验。

## 执行

```bash
export LINGBOT_ROOT=/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va
export PROJECT_ROOT="$LINGBOT_ROOT/code"
export PREPARED_DATA_ROOT="$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42"
export STAGE1_CHECKPOINT="$LINGBOT_ROOT/train_out/robotwin/BASE_RUN/checkpoints/checkpoint_step_15000"
export VLAC_MODEL="$LINGBOT_ROOT/train_out/critic/robotwin/VLAC_CHECKPOINT"
export PART2_RUN_ID="robotwin_part2_$(date +%Y%m%d_%H%M%S)"
export PART2_LOG="$LINGBOT_ROOT/train_out/logs/${PART2_RUN_ID}.log"

bash script/robotwin_part2_all.sh
```

最终只需返回 `$PART2_LOG`。中间输出统一位于：

```text
$LINGBOT_ROOT/train_out/critic/robotwin/part2_rft/
```

可拆开执行：

```bash
bash script/robotwin_part2_prepare.sh
bash script/robotwin_part2_generate_and_select.sh
bash script/run_robotwin_action_only_rft.sh
```

选择器同时生成四个严格同预算数据集：

```text
naive_rft_selected.jsonl
process_rft_selected.jsonl
action_rft_selected.jsonl
dual_rft_selected.jsonl
```

训练 ablation 时只替换 `PSEUDO_JSONL`，其余 checkpoint、steps、global batch 和
seed 必须相同：

```bash
PSEUDO_JSONL="$PART2_ROOT/process_rft_selected.jsonl" \
RUN_ID=process_only_rft \
bash script/run_robotwin_action_only_rft.sh
```

## 验证标准

- `stage1_kinematic_profile.json` 的 `calibration_scope` 必须为
  `stage1_action_only`。
- `stage2_video_contexts.summary.json` 的 `reads_action_column` 必须为 `false`。
- selected 数量必须等于 `stage2_chunk_budget.json`，且每个 task/domain 都相等。
- RFT manifest 中 video/shared 参数为 frozen，只有三个 action-specific module
  可训练。
- 每个方法使用同一 M30 checkpoint、同一 pseudo budget、同一 optimizer steps。
- 最终在未参与训练的 RoboTwin seeds 上评测，每个 task/domain 100 episodes，
  至少 3 个不同 30/20 split，报告 success rate 的均值、标准差和 per-task 结果。

目标是 `Dual-RFT > Base-30`，并尽量接近使用全部真实 action 的 `Base-50`。

## 当前测试状态

本地 CPU 单元测试已覆盖 Kinematic 异常检测、四元数符号不变性、Stage1
provenance、四种选择规则、exact-budget fail-closed 和 70/30 index 逻辑。
真实 Torch/FSDP、官方 checkpoint、VLAC 端到端 smoke 尚需在 H100 服务器执行；
当前本地 SSH 隧道 `127.0.0.1:2222` 在 banner 阶段超时。
