# LingBot-VA RoboTwin Clean 训练与对齐评测

本文档是这个代码快照的主入口。目标是在一台新的多 GPU 服务器上完成：

1. 安装 LingBot-VA 与 RoboTwin 环境；
2. 下载官方基础模型、官方 RoboTwin 模型、完整 Clean+Aug 数据和仿真资产；
3. 先用官方模型完成 32 任务 x 20 episodes 的 Easy/Clean 校准；
4. 只有官方模型成功率达到 85% 后，才启动 Clean-only 训练；
5. 用完全相同的协议评测训练 checkpoint。

仓库保存全部源码和配置，但不包含模型、数据集、Python 环境、RoboTwin 大型资产、checkpoint、日志或结果。

## 1. 推荐目录

把仓库克隆到统一项目根下的 `code/`：

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
git clone <YOUR_GITHUB_REPOSITORY> "$LINGBOT_ROOT/code"
cd "$LINGBOT_ROOT/code"
```

最终目录应为：

```text
$LINGBOT_ROOT/
├── code/                         # 本 GitHub 仓库
├── models/
│   ├── lingbot-va-base/
│   └── lingbot-va-posttrain-robotwin/
├── datasets/
│   └── robotwin-clean-and-aug-lerobot/
│       ├── lerobot_robotwin_eef_clean_50/
│       ├── lerobot_robotwin_eef_aug_500/
│       └── empty_emb.pt
├── train_out/                    # 训练、评测、prompt cache
└── logs/
```

## 2. 软件与硬件

已验证的核心版本：

```text
Ubuntu 22.04
Python 3.10.16
PyTorch 2.9.0
CUDA wheel 12.6
torchvision 0.24.0
torchaudio 2.9.0
transformers 4.55.2
diffusers 0.36.0
LeRobot 0.3.3
SAPIEN 3.0.0b1
MPLib 0.2.1
RoboTwin commit 2eeec322
```

推荐 H100/H200 80GB 或 RTX 5090。训练和正式评测不要同时运行。评测还要求 Vulkan RT 和 OptiX 可用。

创建环境：

```bash
cd "$LINGBOT_ROOT/code"
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install \
  diffusers==0.36.0 transformers==4.55.2 numpy==1.26.4 \
  accelerate websockets einops msgpack opencv-python matplotlib \
  ftfy easydict tqdm "imageio[ffmpeg]" safetensors Pillow \
  modelscope huggingface_hub swanlab scipy wandb lerobot==0.3.3

python -m pip install flash-attn --no-build-isolation
python -m pip install -e .
```

保存环境证据：

```bash
python --version
python -m pip freeze > environment-freeze.txt
nvidia-smi
```

不要从其他服务器复制 FlashAttention、PyTorch3D、cuRobo 的 `.so` 文件；它们必须针对目标服务器的 PyTorch/CUDA 重新编译。

## 3. 必须下载的内容

### 3.1 基础模型

训练初始化使用：

- Hugging Face: `Robbyant/lingbot-va-base`
- 目标：`$LINGBOT_ROOT/models/lingbot-va-base`

```bash
bash script/download_cn.sh \
  lingbot-va-base \
  "$LINGBOT_ROOT/models/lingbot-va-base"
```

训练前必须确认：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["LINGBOT_ROOT"]) / "models/lingbot-va-base/transformer/config.json"
config = json.loads(path.read_text())
config["attn_mode"] = "flex"
path.write_text(json.dumps(config, indent=2) + "\n")
print("base model training config set to flex")
PY
```

### 3.2 官方 RoboTwin 模型

评测器校准使用：

- Hugging Face: `Robbyant/lingbot-va-posttrain-robotwin`
- 目标：`$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin`

```bash
bash script/download_cn.sh \
  lingbot-va-posttrain-robotwin \
  "$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin"
```

该模型的 `transformer/config.json` 必须是 `"attn_mode": "torch"` 或 `"flashattn"`，不能是训练用的 `"flex"`。

### 3.3 下载完整 RoboTwin Clean+Aug 数据

官方数据仓库：

- Hugging Face dataset: `Robbyant/robotwin-clean-and-aug-lerobot`

必须下载完整数据包，同时保留：

```text
lerobot_robotwin_eef_clean_50
lerobot_robotwin_eef_aug_500
empty_emb.pt
```

推荐使用仓库下载脚本：

```bash
bash script/download_cn.sh \
  robotwin-clean-and-aug-lerobot \
  "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
```

也可以直接从 Hugging Face 下载完整 snapshot：

```bash
mkdir -p "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
huggingface-cli download Robbyant/robotwin-clean-and-aug-lerobot \
  --repo-type dataset \
  --local-dir "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
```

完整性检查：

```bash
test -d "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50"
test -d "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500"
test -s "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/empty_emb.pt"
du -sh "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
```

下载范围是完整 Clean+Aug，但本轮训练范围仍严格限定为 Clean：

```text
$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50
```

`lerobot_robotwin_eef_aug_500` 本轮不参与训练，但必须保留，便于后续 Hard/Aug 训练、数据分析及泛化实验。

### 3.4 RoboTwin 仿真资产

仓库包含 RoboTwin 与 cuRobo 源码，不包含约 16GB 的背景、物体和机器人资产。

```bash
bash script/setup_robotwin_eval.sh
```

该脚本安装 RoboTwin 依赖、编译 PyTorch3D/cuRobo、下载资产并执行渲染检查。RoboTwin 基准版本为 `2eeec322`。

不需要 SQL、向量数据库或其他数据库服务。

## 4. Clean 数据完整性审计

训练前运行：

```bash
python script/audit_robotwin_clean_dataset.py \
  "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50" \
  --json-output "$LINGBOT_ROOT/train_out/clean_dataset_audit.json"
```

预期：

```text
50 task repositories
2500 episodes
约 2492 个可训练 segment
DATASET_AUDIT_OK
```

发布数据的当前快照中已知最多 8 个 segment 缺少至少一个相机 latent，因此默认允许 8 个缺失并训练 2492 个有效 segment。若朋友下载到完整镜像，可用 `--allow-missing-latents 0` 做严格审计。

每个 `action_config` 是一个训练 segment。模型会读取：

- 一段预计算的三相机 video latent；
- 同一帧区间的 action；
- `frame_chunk_size=2`；
- 每个 latent frame 对齐 16 个 action；
- 30 维 action 中实际使用双臂位姿与夹爪通道；
- 按训练配置做相对位姿和分位数归一化。

## 5. 先校准评测器

这是硬门槛，不是可选 smoke。它固定：

```text
demo_clean
32 tasks
20 episodes/task
640 episodes total
deterministic audited seed cache
RT CPG1
RT_DENOISER=optix
FAST=0
LOW_RENDER=0
POLICY_CAMERAS_ONLY=0
DEFER_RENDER_UPDATES=0
RECREATE_CAMERAS_EVERY=0
timing=1
resume_partial=1
SKIP_EXISTING=1
no video
```

种子文件已随代码保存：

```text
evaluation/robotwin/seed_cache/demo_clean_32tasks_seed0_n100.json
SHA256=79df07987ad05a0ac99a23cccfc097e93f11a17d5841340ddb0499597fea97d4
```

它由原始 50-task cache 机械筛选而来，未重新采样 seed。原文件也保留在同目录，SHA256 为 `4794de71cd3140d2a415c72e9a246a5c7373080d3ef72b430a873930d028926c`。

使用所有可见 GPU：

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
NGPU=8 bash script/run_robotwin_clean_official_calibration.sh
```

脚本会自动生成 prompt embedding cache、启动 prompt service、运行 32x20、审计结果，并要求：

```text
OFFICIAL_CALIBRATION_PASSED
SR >= 0.85
```

我们在同一评测器的 32x5 校准中得到官方模型 `139/160 = 86.88%`。32x20 是更稳定的正式口径。若低于 85%，立即停止，不要训练或汇报自有 checkpoint；优先检查：

1. H100/Hopper 是否使用 `RT_DENOISER=optix`；
2. 官方模型 `attn_mode` 是否为 `torch`；
3. SAPIEN、Vulkan、OptiX 和资产是否完整；
4. 是否误开 `FAST`、低渲染、camera recreate 或 defer；
5. seed cache SHA256 是否一致；
6. `place_object_scale.arm_tag` 修复是否存在；
7. client/server 是否出现 OOM、DeviceLost 或无进展。

注意：这个 32-task 校准口径用于验证当前实验链路，不等同于论文表格中的完整 50-task 口径。

## 6. Clean-only 训练

主入口：

```bash
bash script/run_robotwin_clean_train_portable.sh
```

默认基线：

```text
base model: lingbot-va-base
clean dataset only: lerobot_robotwin_eef_clean_50
batch_size=1/GPU
global batch=64
LR=1e-5
warmup=10
constant scheduler
activation checkpointing=1
10000 optimizer steps
save at 2000/4000/6000/8000/10000
```

脚本按 GPU 数自动计算 gradient accumulation，以保持 global batch=64：

| GPU 数 | batch/GPU | gradient accumulation | global batch |
|---:|---:|---:|---:|
| 4 | 1 | 16 | 64 |
| 8 | 1 | 8 | 64 |
| 16 | 1 | 4 | 64 |
| 32 | 1 | 2 | 64 |
| 64 | 1 | 1 | 64 |

例如 8 卡：

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
export NGPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 可选 SwanLab；密钥只放环境变量，不得提交到 GitHub。
export SWANLAB_API_KEY='<your-key>'
export LINGBOT_SWANLAB_MODE=online
export LINGBOT_SWANLAB_WORKSPACE='<your-workspace>'

bash script/run_robotwin_clean_train_portable.sh
```

输出：

```text
$LINGBOT_ROOT/train_out/robotwin/<RUN_ID>/
├── run_manifest.txt
├── train.log
├── exit_code
├── swanlab/
└── checkpoints/
    ├── checkpoint_step_2000/
    ├── checkpoint_step_4000/
    ├── checkpoint_step_6000/
    ├── checkpoint_step_8000/
    └── checkpoint_step_10000/
```

按 2492 个有效 segment、global batch=64 估算，10000 optimizer steps 会抽取约 640000 个样本，相当于约 257 个 dataset passes。因此不要默认最后一个 checkpoint 最好，必须评测 2K、4K、6K、8K、10K。

## 7. 对齐评测训练 checkpoint

只有官方校准通过后运行：

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
export NGPU=8

CHECKPOINT_PATH="$LINGBOT_ROOT/train_out/robotwin/<RUN_ID>/checkpoints/checkpoint_step_2000" \
  bash script/run_robotwin_clean_checkpoint_eval.sh
```

脚本会自动读取最近一次通过的官方校准结果；也可以显式指定：

```bash
CALIBRATION_SUMMARY=/path/to/official/summary.json \
CHECKPOINT_PATH=/path/to/checkpoint_step_4000 \
  bash script/run_robotwin_clean_checkpoint_eval.sh
```

批量串行评测：

```bash
for step in 2000 4000 6000 8000 10000; do
  CHECKPOINT_PATH="$LINGBOT_ROOT/train_out/robotwin/<RUN_ID>/checkpoints/checkpoint_step_${step}" \
    NGPU=8 \
    bash script/run_robotwin_clean_checkpoint_eval.sh
done
```

每次评测都会生成 `summary.json`，包括：

- 总 SR；
- 32 个任务逐任务 SR；
- mean/median/P95 episode timing；
- timing episode 1..20 连续性；
- seed 有序、无重复且来自固定缓存；
- `res.json` 与 timing success 一致性。

更完整的异常判断和恢复规则见 [`RoboTwin_Evaluation_Usage.md`](RoboTwin_Evaluation_Usage.md)。

## 8. GitHub 上传前检查

```bash
git status --short
du -sh .
find . -type f -size +50M -print
rg -n --hidden 'SWANLAB_API_KEY=|HF_TOKEN=|api[_-]?key[[:space:]]*=' .
```

必须确认没有提交：

- `.secrets/`；
- 模型和 dataset；
- RoboTwin `assets/`；
- `.venv/`、编译产物；
- `train_out/`、`logs/`、checkpoint；
- SwanLab/Hugging Face token；
- 旧实验结果或视频。
