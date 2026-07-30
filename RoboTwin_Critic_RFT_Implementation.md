# RoboTwin 50+50 Critic 与 Chunk RFT 实施手册

本文档描述完全旁路于现有 WAM 训练、评测、server/client 的第二阶段流程。所有新增
Python 代码位于：

```text
robotwin_critic/two_stage_rft/
```

所有新入口位于：

```text
script/robotwin_two_stage_vlac_prepare.sh
script/robotwin_two_stage_vlac_train.sh
script/robotwin_action_critic_calibrate.sh
script/robotwin_action_critic_eval.sh
script/robotwin_build_chunk_rft_dataset.sh
script/robotwin_build_mixed_rft_view.sh
script/run_robotwin_chunk_rft_portable.sh
```

现有 `wan_va/train.py`、`wan_va/wan_va_server.py` 和 RoboTwin 评测入口均未修改。

## 1. 统一数据口径

每个 task 固定使用同一份 50 Clean + 50 Randomized：

| 用途 | Clean | Randomized | 是否读取真实 action |
|---|---:|---:|---|
| Stage 1 WAM SFT | 30 | 30 | 是 |
| Stage 2 WAM 生成上下文 | 20 | 20 | 否 |
| VLAC Process Critic | 50 | 50 | 不需要 |
| Action Critic 校准 | 30 | 30 | 是，仅 Stage 1 |
| RFT | Stage 2 生成并通过筛选的 chunk | 同左 | 只用 WAM 候选 action |

VLAC 使用这 50+50 的全部视频。`full` 数据准备会把内部 episode holdout 也并入
`train.jsonl`，满足“同一批 50+50 全部参与 VLAC 优化”。原 `val.jsonl` 仍保留，
但此时只能用于训练过程监控，不能作为泛化结果。论文中的泛化指标必须来自未参与
训练的 simulator seeds、场景扰动或 task split。

Stage 2 原始 parquet 中虽然存在 action，RFT pipeline 不允许读取它们作为监督。
这样实验才能支持“只用 30+30 action 标注，通过 WAM 与 critic 利用额外 observation
完成自提升”的结论。

## 2. VLAC Process Critic

### 2.1 数据

`build_vlac_index.py` 直接读取不可变 `split_manifest.json`，为每个样本写入：

```text
task, domain, stage, source_episode_index, output_episode_index,
task_dir, parquet_path, RGB video paths
```

它会先验证每个 task 恰好具有：

```text
Stage 1: 30 Clean + 30 Randomized
Stage 2: 20 Clean + 20 Randomized
Union:   50 Clean + 50 Randomized
```

随后复用既有 `robotwin_critic.vlac_finetune.build_pairs`，从轨迹中抽取
\((S_i,S_j,S_{\mathrm{final}},g)\)，同时生成 forward/reverse pair。

### 2.2 目标

VLAC 输出状态分数 \(u_\phi(S,S_{\mathrm{final}},g)\)。训练使用两个状态的差：

$$
\Delta u_{ij}
=
u_\phi(S_j,S_{\mathrm{final}},g)
-
u_\phi(S_i,S_{\mathrm{final}},g).
$$

正序 pair 应满足 \(\Delta u_{ij}>0\)，倒序 pair 应满足
\(\Delta u_{ij}<0\)，相邻静态 pair 接近 0。模型采用 **VLAC-2B 全参数训练**：

```text
--train_type full
--freeze_vit false
--freeze_aligner false
```

没有 LoRA。

### 2.3 准备与训练

```bash
export PROJECT_ROOT=/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code
export PREPARED_DATA_ROOT=/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/datasets/robotwin-clean-aug-two-stage-seed42

# 首次运行还会物化固定 50+50 hardlink 数据；随后安装依赖、
# 下载 VLAC，并构造 2-task 小数据
bash script/robotwin_two_stage_vlac_prepare.sh smoke

# 10-step 全参数 smoke
bash script/robotwin_two_stage_vlac_train.sh smoke

# 构造全部 50 tasks x 100 trajectories
bash script/robotwin_two_stage_vlac_prepare.sh full

# 正式全参数训练
bash script/robotwin_two_stage_vlac_train.sh full
```

准备脚本可重复使用已经完整下载的模型；只有 `config.json` 不存在时才下载。
smoke 训练完成后，脚本默认把不含 optimizer state 的 eval-only checkpoint 暂存到
`/tmp/robotwin-vlac-eval`，再执行 VOC/VROC 评测，避免共享盘二次加载权重。可通过
`LOCAL_EVAL_MODEL_ROOT` 改变路径；设为空字符串可关闭本地暂存。

## 3. 解析式 Action Critic

Action Critic 不再训练一个容易记住 task 外观的分类器，而是在 Stage 1 真实 action
上校准物理运动统计量。默认从每个 repository 的 `info.json` 推断 action fps，并
要求所有校准数据的 fps 一致；不要把 latent/video 的抽帧 fps 当成 action fps。

对每只机械臂的位置 \(p_t\) 和四元数 \(q_t\)，计算：

$$
v_t = \frac{p_{t+1}-p_t}{\Delta t},
\qquad
\omega_t =
\frac{2\arccos\left(\left|\langle q_{t+1},q_t\rangle\right|\right)}
{\Delta t}.
$$

再计算：

$$
a_t = \frac{v_{t+1}-v_t}{\Delta t},
\qquad
j_t = \frac{a_{t+1}-a_t}{\Delta t}.
$$

角速度同样得到 angular acceleration 和 angular jerk。绝对内积保证
\(q\) 与 \(-q\) 不会产生假的旋转尖峰。夹爪开合是离散事件，只报告 switch rate，
不进入速度、加速度或 jerk 的拒绝条件。

每项统计从 Stage 1 得到 soft quantile \(Q_s\) 和 hard quantile \(Q_h\)。
候选 chunk 的归一化超限为：

$$
e_k(x)=
\frac{\max(x-Q_{s,k},0)}
{\max(Q_{h,k}-Q_{s,k},\mathrm{robust\_scale}_k)}.
$$

综合分数为：

$$
r_{\mathrm{action}}
=
\exp\left(
-\frac{1}{K}\sum_{k=1}^{K}\operatorname{mean}
\left[\min(e_k,2)\right]
\right).
$$

任何 hard threshold 超限都会拒绝候选，防止平均分掩盖单帧剧烈跳变。

```bash
bash script/robotwin_action_critic_calibrate.sh
```

输出：

```text
train_out/critic/robotwin/action_critic/stage1_profile.json
```

只用 Stage 2 action 做离线评测，不参与阈值校准：

```bash
bash script/robotwin_action_critic_eval.sh
```

该评测将真实 segment 作为正样本，构造 position spike、high-frequency jitter、
orientation spike 和 excess-speed 负样本，输出 AUROC、false reject rate、
false accept rate 和各负样本类型指标。它衡量解析式 critic 对明确运动异常的识别能力，
不代表对所有语义错误 action 的完整覆盖。

## 4. 两个 reward 与筛选顺序

候选 chunk 必须依次通过：

1. Action-video consistency：动作和生成视频一致；
2. Process reward：视频相对目标状态产生正向进展；
3. Action reward：动作在速度、加速度、jerk 上合理。

建议先做 hard filtering，再做软排序：

$$
\mathrm{keep}
=
\mathbb{1}[c_{\mathrm{cons}}> \tau_c]
\cdot
\mathbb{1}[r_{\mathrm{process}}> \tau_p]
\cdot
\mathbb{1}[r_{\mathrm{action}}> \tau_a],
$$

$$
r_{\mathrm{total}}
=
\alpha \sigma(\Delta u/T)
+
(1-\alpha)r_{\mathrm{action}}.
$$

Consistency 不通过的样本直接丢弃，不作为 process critic 的负样本。

候选打完 Action Critic 分数后，用以下命令按 context 排序：

```bash
python -m robotwin_critic.two_stage_rft.select_rft_candidates \
  --input /path/to/scored_candidates.jsonl \
  --output /path/to/selected_candidates.jsonl \
  --alpha 0.7 \
  --process-temperature 1.0 \
  --min-process-score 0.0 \
  --min-action-score 0.5 \
  --top-k 1
```

## 5. Chunk RFT 与原训练语义

原 `LatentLeRobotDataset` 的一个 dataset item 不是单个模型 chunk，而是一个
`action_config` 轨迹段。loader 再把该轨迹段按 `frame_chunk_size=2` 做时间打包。

因此本实现采用：

```text
one selected WAM chunk
  -> one short LeRobot episode
  -> one action_config trajectory segment
  -> unchanged LatentLeRobotDataset
  -> normal WAM frame_chunk_size=2 packing
```

`rft_dataset.py` 会：

1. 只接收 consistency、process、action 三关通过的候选；
2. 从 source parquet 截取同长度的 schema/template，不读取 Stage 2 原 action；
3. 用候选 action 完整替换 `action` 列；
4. 重编号 episode/frame/global index；
5. 拷贝并重定位三路生成 latent 的 `frame_ids`；
6. 检查三相机 frame ids 一致；
7. 按原 `_action_post_process` 公式检查 action 数量；
8. 写到新的 RFT root，绝不覆盖源数据；
9. 完成后顺序实例化未改动的 `LatentLeRobotDataset` 并检查 tensor shape。

候选 JSONL 的必需字段：

```json
{
  "task": "adjust_bottle",
  "text": "adjust the bottle",
  "source_repo": "/path/to/stage2/task/repo",
  "source_parquet": "/path/to/episode.parquet",
  "start_frame": 64,
  "end_frame": 145,
  "fps": 30,
  "action_path": "/path/to/generated_actions.npy",
  "latent_paths": {
    "observation.images.cam_high": "/path/to/high.pth",
    "observation.images.cam_left_wrist": "/path/to/left.pth",
    "observation.images.cam_right_wrist": "/path/to/right.pth"
  },
  "consistency": {"accepted": true},
  "process_score": 0.37,
  "action_critic": {"accepted": true, "action_score": 0.91}
}
```

构造：

```bash
SELECTED_JSONL=/path/to/scored_candidates.jsonl \
OUTPUT_ROOT=/path/to/new-rft-dataset \
bash script/robotwin_build_chunk_rft_dataset.sh
```

### 5.1 构造 Stage 1 + RFT 混合视图

正式 RFT 不直接只训练生成数据。先构造不可变混合 view：

```bash
export RFT_DATA_ROOT=/path/to/new-rft-dataset
export MIXED_DATA_ROOT=/path/to/new-stage1-rft-mixed-view
export RFT_TARGET_FRACTION=0.25

bash script/robotwin_build_mixed_rft_view.sh
```

该 view 不复制大文件：

- 每个 source repository 创建独立、极小的 metadata wrapper；
- `data/latents/videos` 使用相对软链接；
- Stage 1 与 RFT 都由原 recursive dataset discovery 找到；
- 根据 `action_config` item 数，而不是 episode 数，计算 RFT 采样占比；
- 必要时重复暴露同一 RFT repository 来接近目标占比；
- 写出 `mixed_view_manifest.json`，记录真实 item 数、重复次数和两个源 manifest SHA。

默认目标 RFT 占比为 25%。由于 repository repetition 是整数，manifest 同时记录
`requested_rft_fraction` 和实际的 `effective_rft_fraction`。

### 5.2 从 Stage 1 checkpoint 做 chunk RFT

```bash
export STAGE1_CHECKPOINT="$LINGBOT_ROOT/train_out/robotwin/<stage1-run>/checkpoints/checkpoint_step_15000"
export MIXED_DATA_ROOT=/path/to/new-stage1-rft-mixed-view

# 只检查 checkpoint、manifest 和原 loader，不启动 GPU
VALIDATE_ONLY=1 bash script/run_robotwin_chunk_rft_portable.sh

# 正式启动；初始协议为 3000 optimizer steps，每 1000 steps 保存
bash script/run_robotwin_chunk_rft_portable.sh
```

正式入口复用原 `run_robotwin_clean_train_portable.sh`，只改变：

```text
initial model = Stage 1 checkpoint_step_15000
dataset = immutable Stage 1 + selected chunk RFT mixed view
train stage = stage2_chunk_rft
optimizer steps = 3000
save steps = 1000,2000,3000
global batch = 64
```

这是第一版对齐协议，不声称 25% 或 3000 steps 已最优。后续 ablation 应固定其他变量，
比较 RFT fraction、RFT steps 和 reward 组合。由于每个生成 chunk 被物化成一个短
`action_config` segment，训练仍走原 loader、action 对齐、temporal packing 和
video/action loss；与长轨迹训练的差别只在跨 segment 历史被重置，因此必须保留
Stage 1 真实轨迹混合，避免上下文分布全部变成短片段。

## 6. 必做实验

| 组别 | WAM action 标注 | Critic 视频池 | Stage 2 |
|---|---:|---:|---|
| Base-30 | 30+30/task | 无 | 无 |
| Base-50 | 50+50/task | 无 | 真实 action SFT |
| Ours-30+RFT | 30+30/task | 同一 50+50/task | critic 筛选的 WAM chunk |
| Random-RFT | 30+30/task | 同一 50+50/task | 随机候选 |
| Process-only | 30+30/task | 同一 50+50/task | 只按 process |
| Process+Action | 30+30/task | 同一 50+50/task | 两 reward |

核心比较是 `Ours-30+RFT` 对 `Base-30`，证明少 action 标注下的提升；
`Base-50` 是监督数据上界，不要求方法必须超过它。所有组使用相同 task、seed、候选数、
optimizer steps 和评测协议。

重点报告：

- RoboTwin success rate 与每 task success；
- Process pair accuracy、macro-F1、Spearman、VOC/VROC；
- Action critic 的 smooth/spike/mismatch AUROC、FPR@95TPR、拒绝类型分布；
- 候选通过率和有效 RFT chunk 数；
- RFT 相对 Base-30 的成功率提升；
- 同等 action 标注量和同等训练 compute 下的提升。

## 7. 当前代码验证记录

本次实现只做 smoke，不启动正式训练。H100 隔离 worktree 中已完成：

```text
robotwin_critic/two_stage_rft unit tests: 8 passed
shell syntax checks: passed
git diff whitespace check: passed
```

真实 RoboTwin segment smoke：

```text
source segment frames: 74
frame stride: 4
packed latent frames: 5
loader latent shape: [48, 5, 24, 20]
loader action shape: [30, 5, 16, 1]
loader action-mask shape: [30, 5, 16, 1]
```

真实混合 view smoke：

```text
Stage 1 items: 50
RFT items: 1
loader discovered total items: 51
one Stage 1 and one RFT sample were both loaded successfully
```

训练 wrapper：

```text
VALIDATE_ONLY=1 passed
initial checkpoint files verified
mixed manifest verified
RFT steps: 3000
save steps: 1000,2000,3000
```

2026-07-30 在 4 张 H100 上完成了 VLAC-2B 全参数 10-step smoke。使用
`smoke_2task` 的 512 个 train pair 和 96 个 validation pair，结果为：

```text
train steps: 10/10
train loss: 3.5114
final validation loss: 1.8333
final validation token accuracy: 0.4289
peak memory: 25.76 GiB/GPU
checkpoint: checkpoint-10
```

首步 loss 为 8.2002，后五步单步 loss 约为 1.68--2.16，说明 forward、backward、
四卡 DDP、validation 和全参数 checkpoint 保存链路均已跑通。checkpoint 包含
4.41 GB 模型权重和 8.82 GB optimizer state。

直接从共享盘加载 checkpoint 的单卡 `evaluate_vlac` 在 15 分钟窗口内未进入 CUDA。
将不含 optimizer state 的 4.2 GB eval-only checkpoint 暂存到节点本地 `/tmp` 后，
同一评测成功完成：

| Metric | VLAC-2B baseline | 10-step full FT |
|---|---:|---:|
| Numeric parse rate | 1.0000 | 1.0000 |
| Pair MAE | 34.1024 | 39.2701 |
| Pair sign accuracy | 0.4219 | 0.5781 |
| Pair macro-F1 | 0.3333 | 0.3678 |
| Pair macro OVR-AUC | 0.6109 | 0.5646 |
| Pair target Spearman | 0.0516 | -0.2451 |
| Mean VOC | 0.6078 | -0.2431 |
| Mean VROC | 0.3700 | 0.9059 |
| Mean antisymmetry MAE | 4.4848 | 2.4515 |

评测规模为 64 pairs 和 3 trajectories。四项运行门槛全部通过：

```text
rgb_manifest_nonempty: true
numeric_output_ok: true
voc_finite: true
vroc_finite: true
smoke_passed: true
```

这组结果只证明 RGB 输入、数值输出、全参数训练、checkpoint 重载和 VOC/VROC
计算链路可用，不证明 10-step critic 已经可部署。虽然 sign accuracy 和
antisymmetry 改善，但 MAE、AUC、Spearman 和 VOC 退化，说明极短 smoke 已偏向局部
标签分布。正式结论必须来自 full 训练后的未见 simulator seed、场景扰动或 task split，
并与未微调 VLAC 在同一评测清单上比较。

旁路训练入口现在会在 `finally` 中显式销毁已初始化的 NCCL process group，降低
多 rank 退出时残留的风险。训练和评测结束后均已确认四张 H100 为 0 MiB，且无残留
训练或评测进程。
