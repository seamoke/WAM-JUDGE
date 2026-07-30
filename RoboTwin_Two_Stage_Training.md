# RoboTwin Clean + Randomized 两阶段训练手册

本文档给出本仓库当前的两阶段 SFT 数据口径和可直接执行的命令。流程严格拆成两部分：

1. **一次性数据准备**：每个 task 分别从 Clean 和 Randomized/Aug 中固定抽取 50 条轨迹，再切成 Stage 1 的 30 条和 Stage 2 的 20 条。
2. **正式 Stage 1 SFT**：只读取 Stage 1，每个 task 使用 30 Clean + 30 Randomized，训练 15,000 optimizer steps，每 3,000 steps 保存一次。

数据准备成功后不应重复运行，也不应修改生成的 `split_manifest.json`。训练可以使用不同的 `RUN_ID` 重跑，但绝不能覆盖旧实验。

---

## 1. 固定实验口径

### 1.1 每个 task 的轨迹划分

| 数据域 | 初始抽取 | Stage 1 | Stage 2 |
|---|---:|---:|---:|
| Clean | 50 | 30 | 20 |
| Randomized/Aug | 50 | 30 | 20 |
| 每 task 合计 | 100 | 60 | 40 |

RoboTwin 共 50 个 task，因此：

| 阶段 | Clean | Randomized/Aug | 总轨迹数 |
|---|---:|---:|---:|
| Stage 1 | \(50 \times 30 = 1500\) | \(50 \times 30 = 1500\) | 3000 |
| Stage 2 | \(50 \times 20 = 1000\) | \(50 \times 20 = 1000\) | 2000 |
| 两阶段合计 | 2500 | 2500 | 5000 |

这里的 Randomized 与数据集目录中的 `aug` 是同一数据域：

```text
lerobot_robotwin_eef_aug_500
```

### 1.2 Stage 1 训练协议

```text
initial model: Robbyant/lingbot-va-base
training data: 50 tasks x (30 clean + 30 randomized)
optimizer steps: 15000
save steps: 3000,6000,9000,12000,15000
learning rate: 1e-5
warmup: 10 optimizer steps
scheduler: constant
target global batch: 64
activation checkpointing: enabled
max_episode_frames filter: effectively disabled
```

global batch 的定义为：

$$
\mathrm{global\ batch}
=
\mathrm{batch/GPU}
\times
\mathrm{GPU\ count}
\times
\mathrm{gradient\ accumulation}.
$$

默认每卡 batch 为 1，因此 4 张 GPU 使用 gradient accumulation 16：

$$
1 \times 4 \times 16 = 64.
$$

---

## 2. 前置目录与文件

先完成 [`auto_pipline_readme.md`](auto_pipline_readme.md) 中的代码克隆、环境安装、模型下载和完整 RoboTwin 数据下载。本文后续假设：

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
cd "$LINGBOT_ROOT/code"
```

目录必须至少包含：

```text
$LINGBOT_ROOT/
├── code/
│   ├── .venv/
│   ├── script/
│   └── wan_va/
├── models/
│   └── lingbot-va-base/
│       └── transformer/
│           ├── config.json
│           └── diffusion_pytorch_model.safetensors
└── datasets/
    └── robotwin-clean-and-aug-lerobot/
        ├── empty_emb.pt
        ├── lerobot_robotwin_eef_clean_50/
        │   └── <50 task repositories>/
        └── lerobot_robotwin_eef_aug_500/
            └── <50 task repositories>/
```

只下载 Clean 不够，因为 Stage 1 和 Stage 2 都需要 Randomized/Aug。

---

## 3. Part A：一次性准备两阶段数据

### 3.1 数据准备做什么

入口：

```text
script/prepare_robotwin_two_stage_dataset.py
```

脚本会：

1. 检查 Clean 与 Randomized 是否具有完全相同的 50 个 task；
2. 对每个 task、每个数据域，使用固定 SHA256 排序从源 episode 中选择 50 条；
3. 将排序靠前的 30 条放入 Stage 1，剩余 20 条放入 Stage 2；
4. 检查两个阶段没有轨迹重叠，并且并集恰好等于选中的 50 条；
5. 重编号各输出 LeRobot repository 的 episode 为连续的 `0..N-1`；
6. 同步 `info.json`、`episodes.jsonl`、可选 episode stats、parquet、视频和 latent；
7. 对大文件使用 hardlink，**不会复制视频和 latent 的数据块**；
8. 写出不可变的 `split_manifest.json` 和完成标记；
9. 完整审计通过后才将临时目录原子重命名为正式输出目录。

抽样排序键为：

```text
sha256(seed:domain:task:episode_index)
```

因此同一数据快照、同一 seed 会得到相同划分。

官方 Clean 与 Aug repository 的目录后缀不同。准备脚本会先把
`task-demo_clean_collect_200-50`、`task-piper_clean_50-50` 和
`task-aloha-agilex_randomized_500-1000` 规范化为同一个 canonical task 名；
少数本身已是 task 名的 Aug repository 保持不变。规范化后仍必须是 50 个任务
一一对应，任何缺失或命名碰撞都会中止处理。

### 3.2 只执行一次

推荐固定输出路径：

```bash
export SOURCE_DATA_ROOT="$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
export PREPARED_DATA_ROOT="$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42"
```

执行：

```bash
cd "$LINGBOT_ROOT/code"
source .venv/bin/activate

python script/prepare_robotwin_two_stage_dataset.py \
  --source-root "$SOURCE_DATA_ROOT" \
  --output-root "$PREPARED_DATA_ROOT" \
  --seed 42 \
  --expected-tasks 50 \
  --per-domain-total 50 \
  --stage1-per-domain 30 \
  --link-mode hardlink \
  --allow-missing-latent-segments 8
```

成功标志：

```text
TWO_STAGE_DATASET_PREPARATION_OK
```

输出结构：

```text
$PREPARED_DATA_ROOT/
├── split_manifest.json
├── PREPARATION_COMPLETE.json
├── stage1/
│   ├── empty_emb.pt
│   ├── lerobot_robotwin_eef_clean_stage1_30/
│   │   └── <50 task repositories>/
│   └── lerobot_robotwin_eef_aug_stage1_30/
│       └── <50 task repositories>/
└── stage2/
    ├── empty_emb.pt
    ├── lerobot_robotwin_eef_clean_stage2_20/
    │   └── <50 task repositories>/
    └── lerobot_robotwin_eef_aug_stage2_20/
        └── <50 task repositories>/
```

预期计数：

```text
tasks: 50
stage1_episodes: 3000
stage2_episodes: 2000
total_output_episodes: 5000
stage1_segments: <由 action_config 实际统计>
stage1_valid_segments: <三相机 latent 完整的实际训练样本数>
```

### 3.3 为什么只处理一次

如果正式输出目录已经存在，脚本会直接拒绝覆盖：

```text
Refusing to overwrite prepared dataset
```

这是设计行为。后续只执行审计：

```bash
python script/prepare_robotwin_two_stage_dataset.py \
  --output-root "$PREPARED_DATA_ROOT" \
  --allow-missing-latent-segments 8 \
  --verify-only
```

成功标志：

```text
TWO_STAGE_DATASET_AUDIT_OK
```

`PREPARATION_COMPLETE.json` 保存 `split_manifest.json` 的 SHA256。训练前会自动再次校验这个 SHA，防止划分被手动修改。

如果准备过程被中断，正式目录不会出现；中间文件保留在：

```text
${PREPARED_DATA_ROOT}.preparing
```

先检查失败原因和磁盘状态，再人工处理该临时目录。不要直接对一个来源不明的半成品开始训练。

### 3.4 hardlink 的要求

默认 `--link-mode hardlink` 要求源数据和输出目录位于同一个文件系统。可用下面的命令检查：

```bash
df -T "$SOURCE_DATA_ROOT" "$(dirname "$PREPARED_DATA_ROOT")"
```

如果两个路径不在同一文件系统，改用：

```bash
--link-mode symlink
```

无论 hardlink 还是 symlink，都不要在训练期间移动或删除官方源数据。

---

## 4. Part B：Stage 1 正式 SFT

### 4.1 训练入口

Stage 1 唯一推荐入口：

```text
script/run_robotwin_stage1_sft_portable.sh
```

该脚本在启动训练前会自动执行 `--verify-only`，然后把训练数据路径固定为：

```text
$PREPARED_DATA_ROOT/stage1
```

`wan_va/dataset/lerobot_latent_dataset.py` 会递归加载该目录中的 50 个 Clean task repository 和 50 个 Randomized task repository。Stage 2 不在这个目录下，因此 Stage 1 不会误读剩余 20+20 条轨迹。

### 4.2 SwanLab

在线记录：

```bash
export SWANLAB_API_KEY='<your-swanlab-api-key>'
export LINGBOT_SWANLAB_MODE=online
export LINGBOT_SWANLAB_WORKSPACE='<your-workspace>'
export LINGBOT_SWANLAB_PROJECT=lingbot-va-robotwin-stage1
```

不要把 API key 写入脚本、文档、Git commit 或日志清单。

也可以先离线运行：

```bash
export LINGBOT_SWANLAB_MODE=offline
```

### 4.3 四卡 H100 示例

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
export PREPARED_DATA_ROOT="$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4
export BATCH_SIZE=1
export RUN_ID="robotwin_clean_aug_stage1_4xh100_global64_15000steps_$(date +%Y%m%d_%H%M%S)"

cd "$LINGBOT_ROOT/code"
source .venv/bin/activate

tmux new -s lingbot-stage1
bash script/run_robotwin_stage1_sft_portable.sh
```

脚本自动设置：

```text
gradient_accumulation = 16
global batch = 64
num_steps = 15000
save_steps = 3000,6000,9000,12000,15000
```

其他 GPU 数也必须精确组成 global batch 64：

| GPU 数 | batch/GPU | gradient accumulation | global batch |
|---:|---:|---:|---:|
| 1 | 1 | 64 | 64 |
| 2 | 1 | 32 | 64 |
| 4 | 1 | 16 | 64 |
| 8 | 1 | 8 | 64 |
| 16 | 1 | 4 | 64 |
| 32 | 1 | 2 | 64 |
| 64 | 1 | 1 | 64 |

### 4.4 训练记录

输出：

```text
$LINGBOT_ROOT/train_out/robotwin/$RUN_ID/
├── run_manifest.txt
├── train.log
├── exit_code
├── swanlab/
└── checkpoints/
    ├── checkpoint_step_3000/
    ├── checkpoint_step_6000/
    ├── checkpoint_step_9000/
    ├── checkpoint_step_12000/
    └── checkpoint_step_15000/
```

`run_manifest.txt` 会记录：

```text
train_stage=stage1
dataset=<PREPARED_DATA_ROOT>/stage1
split_manifest=<PREPARED_DATA_ROOT>/split_manifest.json
split_manifest_sha256=<固定 SHA256>
global_batch=64
num_steps=15000
save_steps=3000,6000,9000,12000,15000
```

这使每个 checkpoint 都可以反查到唯一的数据划分。

### 4.5 训练期间监控

另开 shell：

```bash
export RUN_ROOT="$LINGBOT_ROOT/train_out/robotwin/$RUN_ID"

tail -n 100 "$RUN_ROOT/train.log"
cat "$RUN_ROOT/run_manifest.txt"
nvidia-smi
pgrep -af "torch.distributed.run|wan_va.train"
```

错误扫描：

```bash
rg -n "Traceback|CUDA out of memory|NCCL|NaN|Inf|Killed|No space left" \
  "$RUN_ROOT/train.log"
```

健康训练必须满足：

- 所有 rank 存活；
- step 持续增长；
- loss、gradient norm 和 learning rate 都是有限值；
- GPU 显存和计算负载稳定；
- 到达保存步数后 checkpoint 完整落盘；
- 没有另一个训练进程读写同一 `RUN_ID`。

---

## 5. Stage 1 完成审计

```bash
export RUN_ROOT="$LINGBOT_ROOT/train_out/robotwin/$RUN_ID"

cat "$RUN_ROOT/exit_code"
tail -n 50 "$RUN_ROOT/train.log"
```

必须满足：

```text
exit_code = 0
日志到达 15000/15000
TRAIN_DONE rc=0
```

检查五个 checkpoint：

```bash
for step in 3000 6000 9000 12000 15000; do
  ckpt="$RUN_ROOT/checkpoints/checkpoint_step_${step}"
  test -d "$ckpt"
  test -s "$ckpt/transformer/diffusion_pytorch_model.safetensors"
  test -s "$ckpt/transformer/config.json"
  du -sh "$ckpt"
done
```

再次审计数据并核对训练清单中的 SHA：

```bash
python "$LINGBOT_ROOT/code/script/prepare_robotwin_two_stage_dataset.py" \
  --output-root "$PREPARED_DATA_ROOT" \
  --allow-missing-latent-segments 8 \
  --verify-only

cat "$RUN_ROOT/run_manifest.txt"
sha256sum "$PREPARED_DATA_ROOT/split_manifest.json"
```

---

## 6. Stage 2 的边界

本次代码已经一次性准备好 Stage 2 数据：

```text
$PREPARED_DATA_ROOT/stage2
```

每个 task 含：

```text
20 clean + 20 randomized
```

Stage 2 与 Stage 1 没有 episode 重叠。但是本次需求只定义了 Stage 1 baseline 的正式训练协议，**尚未定义 Stage 2 应从哪个 Stage 1 checkpoint 初始化、训练多少步、学习率是否重置以及保存频率**。在这些实验变量明确前，不应自行启动 Stage 2。

---

## 7. 评测

Stage 1 checkpoint 的 Easy 对齐评测仍按：

```text
50 tasks x 10 episodes
```

先使用官方 `Robbyant/lingbot-va-posttrain-robotwin` 校准同一评测器并达到至少 85%，再评测 Stage 1 checkpoint。完整命令、固定 seed、RT/OptiX 参数、断点恢复和结果审计见：

- [`auto_pipline_readme.md`](auto_pipline_readme.md)
- [`RoboTwin_Evaluation_Usage.md`](RoboTwin_Evaluation_Usage.md)

Randomized/Aug 是训练数据域，不等于把 Easy 评测协议自动改为 Hard；如需 Hard 评测，必须按评测文档单独准备 `demo_randomized` seed 和协议。
