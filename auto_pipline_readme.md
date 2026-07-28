# LingBot-VA RoboTwin 从零训练与对齐评测完整手册

本文档是本仓库的**主执行入口**。一个第一次接触本项目的人，只需要按本文档从上到下执行，就可以完成：

1. 克隆代码并建立约定目录；
2. 安装 LingBot-VA 训练环境和 RoboTwin 评测环境；
3. 下载两个官方模型、完整 RoboTwin Clean+Aug 数据集和仿真资产；
4. 审计 50 个 Clean 任务、2500 条轨迹及预计算 latent；
5. 先用官方 RoboTwin 模型校准评测器，确认全部 50 个 Clean 任务成功率不低于 85%；
6. 仅使用 Clean 数据从 `lingbot-va-base` 开始训练；
7. 运行 20,000 optimizer steps，每 5,000 steps 保存 checkpoint；
8. 用完全相同的协议评测 5K/10K/15K/20K checkpoint；
9. 审计每个任务的成功率、seed、timing、墙钟时间和异常恢复记录。

仓库地址：

```text
https://github.com/seamoke/WAM-JUDGE
```

仓库只保存代码、配置、固定 seed 和文档，**不包含**模型、完整数据集、Python 环境、RoboTwin 大型资产、checkpoint、日志、评测结果或密钥。

---

## 0. 先读这一节：口径、入口和验收标准

### 0.1 本次实验的固定范围

| 项目 | 固定设置 |
|---|---|
| 训练初始化 | `Robbyant/lingbot-va-base` |
| 训练数据 | 只使用 `lerobot_robotwin_eef_clean_50` |
| 下载范围 | 完整 `robotwin-clean-and-aug-lerobot`，Clean 和 Aug 都下载 |
| 训练步数 | 20,000 optimizer steps |
| 保存步数 | 5,000 / 10,000 / 15,000 / 20,000 |
| 单卡 batch | 1 |
| 目标 global batch | 64 |
| 学习率 | `1e-5` |
| scheduler | constant |
| warmup | 10 optimizer steps |
| 正式 Easy 评测 | 50 tasks x 20 episodes（共 1000 episodes） |
| 官方校准门槛 | `lingbot-va-posttrain-robotwin` 的 SR >= 85% |
| 自有 checkpoint 评测 | 必须在官方校准通过后执行 |

正式 Easy 评测覆盖的 50 个任务与 `lerobot_robotwin_eef_clean_50` 的训练任务集合一致，不再使用旧实验中的 32-task 子集。官方模型校准和所有自有 checkpoint 必须使用同一 50-task seed cache、同一任务顺序和相同的每任务 episode 数。

这里的“step”是 optimizer step，不是单条轨迹，也不是单个 micro-batch。实际 global batch 为：

$$
\mathrm{global\ batch}
=
\mathrm{batch/GPU}
\times
\mathrm{GPU\ count}
\times
\mathrm{gradient\ accumulation}.
$$

例如 4 张 GPU、每卡 batch 1、gradient accumulation 16：

$$
1 \times 4 \times 16 = 64.
$$

### 0.2 本次实验只使用这些入口

不要直接套用上游 README 中面向通用场景的旧命令。本次对齐实验使用：

```text
训练：
script/run_robotwin_clean_train_portable.sh

任意兼容 GPU 数、global batch 64、20K ZIP 基线训练 + checkpoint 差分 watcher：
script/run_robotwin_clean_zipbaseline_global64_20k_with_delta_audit.sh

checkpoint 与 base 权重差分：
script/compare_checkpoint_to_base.py

官方模型校准：
script/run_robotwin_clean_official_calibration.sh

自有 checkpoint 评测：
script/run_robotwin_clean_checkpoint_eval.sh

数据审计：
script/audit_robotwin_clean_dataset.py

评测结果审计：
script/audit_robotwin_clean_eval.py
```

这些 portable 脚本负责检查路径、模型 attention mode、global batch、固定 seed、正式渲染参数和结果完整性。正式启动优先使用带 delta audit 的 global64 包装器；`run_robotwin_clean_train_portable.sh` 是不启动 watcher 的通用入口。

### 0.3 最终验收标准

在宣布实验完成前，必须同时满足：

```text
[ ] 完整 Clean+Aug 数据已下载
[ ] Clean 数据审计为 DATASET_AUDIT_OK
[ ] RoboTwin Vulkan/RT 渲染检查通过
[ ] 官方模型完成 50x20 且 SR >= 85%
[ ] 训练日志到达 20000/20000
[ ] exit_code 内容为 0
[ ] 日志包含 TRAIN_DONE rc=0
[ ] 恰好存在 4 个目标 checkpoint
[ ] 每个 checkpoint 的模型权重和 config.json 非空
[ ] 每个自有 checkpoint 都使用同一 50x20 协议评测
[ ] 每个 summary.json 均通过 seed/timing/res 严格审计
```

### 0.4 其他文档分别解决什么问题

本文档包含完整可执行流程。遇到需要深入理解的部分，再阅读：

- [`README.md`](README.md)：LingBot-VA 原始项目介绍、模型结构、通用训练和推理背景。
- [`RoboTwin_Evaluation_Usage.md`](RoboTwin_Evaluation_Usage.md)：正式评测协议、Easy/Hard 定义、断点恢复、GPU holder 和结果审计原则。
- [`INSTALL.md`](INSTALL.md)：本代码快照的最小安装说明。
- [`MIGRATION_MANIFEST.md`](MIGRATION_MANIFEST.md)：从原服务器整理到本仓库时保留和排除的内容。
- [`PACKAGE_CONTENTS.txt`](PACKAGE_CONTENTS.txt)：仓库文件清单摘要。
- [`SOURCE_FILES.sha256`](SOURCE_FILES.sha256)：发布时源码文件的 SHA256 清单。

如本文档与上游 `README.md` 的通用示例冲突，以本文档及上述 portable 脚本为准。

---

## 1. 机器要求与资源预算

### 1.1 操作系统和 GPU

已验证参考环境：

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

推荐：

- 训练：4 张或更多 H100/H200 80GB；
- 评测：支持 Vulkan ray tracing 和 OptiX 的 NVIDIA GPU；
- 训练和正式评测不要同时运行；
- 同一轮分布式训练应使用同型号 GPU。

H100/H200 上 CUDA 训练正常不代表 RoboTwin ray-tracing 渲染一定正常。正式评测必须先用官方模型校准。

### 1.2 磁盘、内存和网络

建议最低准备：

```text
可用磁盘：至少 1.5 TB，推荐 2 TB
CPU 内存：至少 128 GB，推荐 256 GB
本地 SSD/NVMe：推荐用于环境和临时编译
共享盘：可用于模型、完整数据集、checkpoint 和结果
```

完整 `robotwin-clean-and-aug-lerobot` 是主要空间占用。下载前执行：

```bash
df -h
df -ih
```

不要把 Hugging Face cache 放在容量很小的系统盘。

### 1.3 驱动与 CUDA 编译工具

PyTorch CUDA wheel 自带运行时，但 FlashAttention、PyTorch3D 和 cuRobo 可能需要本机 CUDA toolkit。先检查：

```bash
nvidia-smi
nvcc --version
which gcc
which g++
```

建议驱动支持 CUDA 12.x，并保证 `nvcc`、驱动和 PyTorch 的 CUDA ABI 可兼容。不要从另一台服务器复制 `.so` 文件。

---

## 2. 克隆仓库并建立固定目录

### 2.1 推荐目录结构

所有脚本默认代码位于项目根的 `code/` 子目录：

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
├── train_out/
├── logs/
└── .cache/
```

### 2.2 克隆和初始化

选择一个有足够空间的位置：

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
mkdir -p "$LINGBOT_ROOT"

git clone https://github.com/seamoke/WAM-JUDGE.git "$LINGBOT_ROOT/code"
cd "$LINGBOT_ROOT/code"

mkdir -p \
  "$LINGBOT_ROOT/models" \
  "$LINGBOT_ROOT/datasets" \
  "$LINGBOT_ROOT/train_out" \
  "$LINGBOT_ROOT/logs" \
  "$LINGBOT_ROOT/.cache/huggingface"

export HF_HOME="$LINGBOT_ROOT/.cache/huggingface"
```

以后每次开新 shell 都至少执行：

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
export CODE_ROOT="$LINGBOT_ROOT/code"
export HF_HOME="$LINGBOT_ROOT/.cache/huggingface"
cd "$CODE_ROOT"
```

### 2.3 核验代码快照

```bash
cd "$LINGBOT_ROOT/code"
git rev-parse HEAD
git status --short
test -f auto_pipline_readme.md
test -f script/run_robotwin_clean_train_portable.sh
test -f script/run_robotwin_clean_official_calibration.sh
test -f evaluation/robotwin/seed_cache/demo_clean_seed0_n100.json
```

可选：核验发布源码清单。该清单不包含下载后的模型和数据：

```bash
sha256sum -c SOURCE_FILES.sha256
```

如果朋友在仓库内主动改过文件，这一步会报告差异，应保存 `git diff` 作为实验记录。

---

## 3. 安装系统依赖

以下命令适用于 Ubuntu 22.04：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10 python3.10-dev python3.10-venv \
  git git-lfs curl wget unzip rsync \
  build-essential ninja-build cmake pkg-config \
  ffmpeg \
  libvulkan1 mesa-vulkan-drivers vulkan-tools \
  libegl1 libgl1 libgl1-mesa-glx libosmesa6

git lfs install
```

检查：

```bash
python3.10 --version
git --version
ffmpeg -version | head
vulkaninfo --summary
```

如果 `vulkaninfo` 在没有 display 的服务器上出现 surface 相关提示，不代表一定失败；最终以第 7 节的 SAPIEN/RoboTwin 渲染测试为准。

---

## 4. 创建 Python 环境

### 4.1 创建 venv

```bash
cd "$LINGBOT_ROOT/code"
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install ninja packaging
```

### 4.2 安装固定 PyTorch

```bash
python -m pip install \
  torch==2.9.0 \
  torchvision==0.24.0 \
  torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

确认所有 GPU 可见：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
PY
```

### 4.3 安装 LingBot-VA 依赖

先安装锁定依赖：

```bash
python -m pip install \
  diffusers==0.36.0 \
  transformers==4.55.2 \
  numpy==1.26.4 \
  accelerate \
  websockets \
  einops \
  msgpack \
  opencv-python \
  matplotlib \
  ftfy \
  easydict \
  tqdm \
  "imageio[ffmpeg]" \
  safetensors \
  Pillow \
  modelscope \
  huggingface_hub \
  swanlab \
  scipy \
  wandb \
  lerobot==0.3.3
```

安装 FlashAttention。编译可能持续较久：

```bash
MAX_JOBS="$(nproc)" python -m pip install flash-attn --no-build-isolation
```

最后只安装本仓库本身，**不要让 editable install 再解析并升级依赖**：

```bash
python -m pip install -e . --no-deps
```

检查关键版本：

```bash
python - <<'PY'
import diffusers
import flash_attn
import lerobot
import numpy
import torch
import transformers

print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
print("flash_attn", flash_attn.__version__)
print("lerobot", getattr(lerobot, "__version__", "unknown"))
PY
```

保存环境证据：

```bash
python --version > "$LINGBOT_ROOT/logs/python-version.txt"
python -m pip freeze > "$LINGBOT_ROOT/logs/environment-freeze.txt"
nvidia-smi > "$LINGBOT_ROOT/logs/nvidia-smi-before-setup.txt"
```

---

## 5. 下载两个模型

必须下载：

| 用途 | Hugging Face ID | 本地目录 |
|---|---|---|
| 训练初始化 | `Robbyant/lingbot-va-base` | `$LINGBOT_ROOT/models/lingbot-va-base` |
| 官方评测校准 | `Robbyant/lingbot-va-posttrain-robotwin` | `$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin` |

### 5.1 使用仓库脚本下载

脚本先尝试 ModelScope，失败后尝试 Hugging Face mirror，并支持 snapshot 续传：

```bash
cd "$LINGBOT_ROOT/code"
source .venv/bin/activate

bash script/download_cn.sh \
  lingbot-va-base \
  "$LINGBOT_ROOT/models/lingbot-va-base"

bash script/download_cn.sh \
  lingbot-va-posttrain-robotwin \
  "$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin"
```

### 5.2 Hugging Face 直连备用命令

若镜像不可用：

```bash
huggingface-cli download Robbyant/lingbot-va-base \
  --local-dir "$LINGBOT_ROOT/models/lingbot-va-base"

huggingface-cli download Robbyant/lingbot-va-posttrain-robotwin \
  --local-dir "$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin"
```

新版 CLI 也可使用：

```bash
hf download Robbyant/lingbot-va-base \
  --local-dir "$LINGBOT_ROOT/models/lingbot-va-base"
```

下载中断时，重新运行完全相同的命令，不要删除已下载目录。

### 5.3 检查模型完整性

```bash
test -s "$LINGBOT_ROOT/models/lingbot-va-base/transformer/config.json"
test -s "$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin/transformer/config.json"

find "$LINGBOT_ROOT/models/lingbot-va-base" -name "*.safetensors" -size +0 -print
find "$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin" -name "*.safetensors" -size +0 -print

du -sh "$LINGBOT_ROOT/models/lingbot-va-base"
du -sh "$LINGBOT_ROOT/models/lingbot-va-posttrain-robotwin"
```

### 5.4 配置 attention mode

训练使用的基础模型必须为 `"flex"`：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["LINGBOT_ROOT"]) / "models/lingbot-va-base/transformer/config.json"
config = json.loads(path.read_text(encoding="utf-8"))
config["attn_mode"] = "flex"
path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(path, "attn_mode=flex")
PY
```

官方评测模型必须保持 `"torch"` 或 `"flashattn"`，不能改为 `"flex"`：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["LINGBOT_ROOT"]) / "models/lingbot-va-posttrain-robotwin/transformer/config.json"
mode = json.loads(path.read_text(encoding="utf-8")).get("attn_mode")
print(path, "attn_mode=", mode)
assert mode in {"torch", "flashattn"}
PY
```

训练和评测使用两个独立模型目录，因此不要在评测前来回修改同一个目录。

---

## 6. 下载完整 RoboTwin Clean+Aug 数据集

### 6.1 必须下载完整数据

数据集 ID：

```text
Robbyant/robotwin-clean-and-aug-lerobot
```

必须完整保留：

```text
$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/
├── lerobot_robotwin_eef_clean_50/
├── lerobot_robotwin_eef_aug_500/
└── empty_emb.pt
```

虽然本次训练只读取 `clean_50`，仍必须下载和保留 `aug_500`，用于之后的 Hard/Aug 训练、统计和泛化评测。不要只下载 Clean 子目录。

### 6.2 推荐下载命令

```bash
cd "$LINGBOT_ROOT/code"
source .venv/bin/activate

bash script/download_cn.sh \
  robotwin-clean-and-aug-lerobot \
  "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
```

Hugging Face 直连备用：

```bash
huggingface-cli download Robbyant/robotwin-clean-and-aug-lerobot \
  --repo-type dataset \
  --local-dir "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
```

这是长时间、大容量下载。建议放在 `tmux` 中：

```bash
tmux new -s robotwin-download
```

进入 tmux 后运行下载命令；使用 `Ctrl-b d` 退出但保留任务，重新进入：

```bash
tmux attach -t robotwin-download
```

### 6.3 完整性检查

```bash
test -d "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50"
test -d "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500"
test -s "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/empty_emb.pt"

find "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50" \
  -path "*/meta/info.json" | wc -l

du -sh "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
df -h "$LINGBOT_ROOT"
```

第一条 `find ... | wc -l` 应输出 `50`。

### 6.4 Clean 数据严格审计

```bash
mkdir -p "$LINGBOT_ROOT/train_out"

python script/audit_robotwin_clean_dataset.py \
  "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50" \
  --json-output "$LINGBOT_ROOT/train_out/clean_dataset_audit.json"
```

预期核心结果：

```text
task_repositories: 50
episodes: 2500
segments: 约 2500
valid_segments: 约 2492
DATASET_AUDIT_OK
```

当前发布快照最多允许 8 个 segment 缺少至少一个相机 latent。若下载源声称已经修复，可执行：

```bash
python script/audit_robotwin_clean_dataset.py \
  "$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50" \
  --allow-missing-latents 0
```

如果缺失超过允许值，不要训练。重新运行 snapshot 下载，确认 Git LFS/大文件没有被替换成指针文本。

### 6.5 训练样本到底如何构造

Clean 数据包含 50 个任务，每个任务 50 条 episode，共 2500 条轨迹。训练不是简单地“一条轨迹等于一个 optimizer step”：

1. 每个 episode 的 `action_config` 定义一个或多个训练 segment；
2. 每个 segment 对应一个帧区间；
3. 三个相机必须都有该区间的预计算 video latent；
4. `frame_chunk_size=2`；
5. 每个 latent frame 对齐 `action_per_frame=16` 个 action；
6. 原始 action 维度为 30，训练按配置选择双臂位姿和夹爪相关通道；
7. action 使用固定 quantile 统计做归一化；
8. DataLoader 会在 optimizer steps 中反复采样这些 segment。

关键实现：

- [`wan_va/dataset/lerobot_latent_dataset.py`](wan_va/dataset/lerobot_latent_dataset.py)
- [`wan_va/configs/va_robotwin_cfg.py`](wan_va/configs/va_robotwin_cfg.py)
- [`wan_va/configs/va_robotwin_train_cfg.py`](wan_va/configs/va_robotwin_train_cfg.py)
- [`README.md`](README.md) 的 Post-Training 章节。

按约 2492 个有效 segment 和 global batch 64 估算：

```text
20000 optimizer steps x 64 = 1280000 segment draws
1280000 / 2492 ≈ 514 dataset-equivalent passes
```

这里的 pass 是按 segment 数量估算，不等价于完整视频轨迹被逐帧完整遍历 514 次。因此不能仅凭“epoch 数”判断最好 checkpoint，必须评测 5K/10K/15K/20K。

---

## 7. 安装 RoboTwin 仿真和渲染环境

仓库已经 vendor 了 RoboTwin 和 cuRobo 源码，但没有提交约 16GB 的背景、物体、机器人等资产。

### 7.1 一键安装

```bash
cd "$LINGBOT_ROOT/code"
source .venv/bin/activate

export ROBOTWIN_DIR="$LINGBOT_ROOT/code/third_party/RoboTwin"
bash script/setup_robotwin_eval.sh
```

该脚本会：

1. 安装 Vulkan/GL 系统包；
2. 使用 RoboTwin commit `2eeec322` 的 vendored 源码；
3. 安装 SAPIEN、MPLib 等依赖；
4. 为当前 PyTorch/CUDA 编译 PyTorch3D 和 cuRobo；
5. 下载 RoboTwin 仿真资产；
6. 运行 SAPIEN render test；
7. 写出 `script/.robotwin_eval_env`。

编译不能跨服务器复制。更完整的协议说明见 [`RoboTwin_Evaluation_Usage.md`](RoboTwin_Evaluation_Usage.md)。

### 7.2 安装完成检查

```bash
source "$LINGBOT_ROOT/code/.venv/bin/activate"
source "$LINGBOT_ROOT/code/script/.robotwin_eval_env"

python - <<'PY'
import mplib
import sapien
import torch

print("torch", torch.__version__)
print("sapien", sapien.__version__)
print("mplib", mplib.__version__)
PY

vulkaninfo --summary
test -d "$LINGBOT_ROOT/code/third_party/RoboTwin/assets/embodiments"
test -d "$LINGBOT_ROOT/code/third_party/RoboTwin/assets/objects"
```

### 7.3 必须存在的评测修复

本仓库已包含并应保留：

- partial resume 按 timing 中真实 seed 恢复；
- seed cache 缺可靠 timing 时拒绝盲目恢复；
- `place_object_scale.arm_tag` 在首次成功判断前初始化；
- 动态 shard 文件锁；
- `SKIP_EXISTING=1`；
- 每 episode timing；
- 同 RUN_ID 安全恢复。

不要只复制上游 RoboTwin 的单个启动脚本覆盖本仓库版本。

---

## 8. 先用官方模型校准评测器

### 8.1 为什么必须先校准

如果官方 `lingbot-va-posttrain-robotwin` 在当前机器上都不能达到至少 85%，自有 checkpoint 的低成功率不能说明训练失败。常见原因是：

- Vulkan/OptiX 渲染不一致；
- 官方模型 attention mode 错误；
- 资产或相机配置缺失；
- seed 或任务定义不一致；
- 错开了 FAST、LOW_RENDER、DEFER、RECREATE；
- server/client、GPU mapping 或恢复逻辑异常。

因此官方校准是硬门槛，不是可选 smoke。

### 8.2 固定校准协议

```text
task_config=demo_clean
50 tasks
20 episodes/task
1000 episodes total
deterministic audited seed cache
RT CPG1
RT_DENOISER=optix
FAST=0
LOW_RENDER=0
POLICY_CAMERAS_ONLY=0
DEFER_RENDER_UPDATES=0
RECREATE_CAMERAS_EVERY=0
WAN_VA_ENABLE_OFFLOAD=0
WAN_VA_OFFLOAD_VAE=0
WAN_VA_OFFLOAD_TEXT_ENCODER=1
timing=1
resume_partial=1
SKIP_EXISTING=1
no video
```

固定 seed：

```text
evaluation/robotwin/seed_cache/demo_clean_seed0_n100.json
SHA256=4794de71cd3140d2a415c72e9a246a5c7373080d3ef72b430a873930d028926c
```

核验：

```bash
sha256sum evaluation/robotwin/seed_cache/demo_clean_seed0_n100.json
```

### 8.3 确认没有训练或旧评测

```bash
pgrep -af "torch.distributed.run|wan_va.train|run_robotwin_eval|run_server_ckpt|eval_polict_client_openpi" || true
nvidia-smi
```

正式评测前应停止训练；不要在同一批 GPU 上同时跑训练和评测。

### 8.4 启动官方 50x20 校准

例如 4 卡：

```bash
cd "$LINGBOT_ROOT/code"
source .venv/bin/activate
source script/.robotwin_eval_env

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4

tmux new -s lingbot-official-calibration
NGPU=4 bash script/run_robotwin_clean_official_calibration.sh
```

离开 tmux：`Ctrl-b d`。重新进入：

```bash
tmux attach -t lingbot-official-calibration
```

### 8.5 校准输出和通过条件

默认输出：

```text
$LINGBOT_ROOT/train_out/robotwin-clean-calibration/<RUN_ID>/
├── manifest.txt
├── results/
├── summary.json
└── exit_code

$LINGBOT_ROOT/logs/robotwin-clean-eval/<RUN_ID>/
```

成功结束应看到：

```text
AUDIT_OK
OFFICIAL_CALIBRATION_PASSED
SR >= 0.85
```

最新通过的 summary 路径会写入：

```text
$LINGBOT_ROOT/train_out/robotwin-clean-calibration/LATEST_PASSED
```

旧的 32x5 校准结果 `139/160 = 86.88%` 只可作为历史排障参考，不能再作为当前协议的通过证据。正式校准必须完整运行 50x20；结果允许有抽样波动，但必须达到脚本门槛 85%。

### 8.6 低于 85% 时怎么办

立即停止，不要训练或评测自有 checkpoint。依次检查：

```bash
cat "$LINGBOT_ROOT/train_out/robotwin-clean-calibration/LATEST_PASSED" 2>/dev/null || true
find "$LINGBOT_ROOT/logs/robotwin-clean-eval" -type f -name "*.log" -print
rg -n "Traceback|CUDA out of memory|DeviceLost|AttributeError|timeout|no.progress" \
  "$LINGBOT_ROOT/logs/robotwin-clean-eval"
```

然后核对：

1. 官方模型 `attn_mode` 是 `torch` 或 `flashattn`；
2. `ROBOTWIN_RT_DENOISER=optix`；
3. 固定 seed SHA 正确；
4. 50 个任务完整；
5. assets 完整；
6. 没有 `FAST=1`、`LOW_RENDER=1` 或 `RECREATE_CAMERAS_EVERY=1`；
7. `place_object_scale` 修复存在；
8. server/client 持续落盘。

详细判据见 [`RoboTwin_Evaluation_Usage.md`](RoboTwin_Evaluation_Usage.md)。

---

## 9. 启动 Clean-only ZIP 基线训练

### 9.1 默认训练配置

`script/run_robotwin_clean_train_portable.sh` 默认就是本次 ZIP 基线：

```text
model: lingbot-va-base
dataset: lerobot_robotwin_eef_clean_50
batch_size: 1/GPU
target global batch: 64
learning_rate: 1e-5
warmup_steps: 10
lr_scheduler: constant
activation_checkpointing: 1
max_episode_frames: effectively disabled
num_steps: 20000
save_steps: 5000,10000,15000,20000
SwanLab: enabled, offline by default
```

脚本会按 GPU 数自动计算 gradient accumulation：

| GPU 数 | batch/GPU | gradient accumulation | global batch |
|---:|---:|---:|---:|
| 1 | 1 | 64 | 64 |
| 2 | 1 | 32 | 64 |
| 4 | 1 | 16 | 64 |
| 8 | 1 | 8 | 64 |
| 16 | 1 | 4 | 64 |
| 32 | 1 | 2 | 64 |
| 64 | 1 | 1 | 64 |

当前 DDP 实现要求所有 rank 的单卡 batch 和 accumulation 相同，因此精确 global batch 64 支持的 GPU 数为 64 的正整数因子，例如 1/2/4/8/16/32/64。3、6、10 等 GPU 数无法在 `batch/GPU=1` 和整数 accumulation 下精确组成 64，正式包装器会拒绝启动，不会静默改变实验协议。

### 9.2 SwanLab 登录

不要把 key 写入脚本、文档或 Git：

```bash
export SWANLAB_API_KEY='<your-swanlab-api-key>'
export LINGBOT_SWANLAB_MODE=online
export LINGBOT_SWANLAB_WORKSPACE='<your-workspace>'
export LINGBOT_SWANLAB_PROJECT=lingbot-va-robotwin
```

也可以离线运行：

```bash
export LINGBOT_SWANLAB_MODE=offline
```

### 9.3 Global-batch-64 标准启动命令

先确认 GPU 空闲：

```bash
nvidia-smi
pgrep -af "torch.distributed.run|wan_va.train" || true
```

设置唯一 RUN_ID：

```bash
export RUN_ID="robotwin_clean_zipbaseline_4xh100_b1_ga16_global64_constant_20000steps_ckpt5000_$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPU=4
export BATCH_SIZE=1
export TARGET_GLOBAL_BATCH=64
export NUM_STEPS=20000
export SAVE_INTERVAL=5000
export WARMUP_STEPS=10
export LR_SCHEDULER=constant
export ACTIVATION_CHECKPOINTING=1
```

推荐启动方式会同时拉起一个**独立 CPU watcher**。watcher 每 60 秒检查 checkpoint，确认模型文件连续两次大小和修改时间稳定后，才比较该 checkpoint 与原始 `lingbot-va-base`：

```bash
cd "$LINGBOT_ROOT/code"
source .venv/bin/activate

tmux new -s lingbot-clean-20k
bash script/run_robotwin_clean_zipbaseline_global64_20k_with_delta_audit.sh
```

该包装器会：

1. 根据 `NGPU` 自动设置 `gradient_accumulation=64/NGPU`；
2. 启动 ZIP 基线 20K 训练并固定 global batch 64；
3. 写出训练 PID；
4. 在 `.venv` 中启动独立 checkpoint watcher；
5. 分别审计 5K/10K/15K/20K；
6. 训练完成后等待四个差分审计全部结束；
7. 分别保存训练和 watcher 的退出码。

如果不需要自动差分 watcher，才使用通用入口：

```bash
bash script/run_robotwin_clean_train_portable.sh
```

不要把训练入口改为裸 `run_robotwin_train.sh`，否则可能丢失路径、global batch、保存步数和覆盖保护。

### 9.4 训练输出

```text
$LINGBOT_ROOT/train_out/robotwin/<RUN_ID>/
├── run_manifest.txt
├── train.log
├── exit_code
├── train_exit_code
├── swanlab/
├── checkpoint_delta_vs_base/
│   ├── watcher.log
│   ├── watcher.pid
│   ├── watcher_exit_code
│   ├── watch_status.json
│   ├── checkpoint_step_5000_vs_base.json
│   ├── checkpoint_step_5000_vs_base.txt
│   └── ...
└── checkpoints/
    ├── checkpoint_step_5000/
    ├── checkpoint_step_10000/
    ├── checkpoint_step_15000/
    └── checkpoint_step_20000/
```

脚本拒绝覆盖已存在的 `OUT`。重新开始实验必须使用新 RUN_ID；不要删除旧结果来复用名字。

### 9.5 训练期间检查

另开 shell：

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
export RUN_ID='<实际 RUN_ID>'
export RUN_ROOT="$LINGBOT_ROOT/train_out/robotwin/$RUN_ID"

tail -n 100 "$RUN_ROOT/train.log"
cat "$RUN_ROOT/run_manifest.txt"
cat "$RUN_ROOT/checkpoint_delta_vs_base/watch_status.json" 2>/dev/null || true
tail -n 50 "$RUN_ROOT/checkpoint_delta_vs_base/watcher.log" 2>/dev/null || true
nvidia-smi
pgrep -af "torch.distributed.run|wan_va.train"
```

错误扫描：

```bash
rg -n "Traceback|CUDA out of memory|NCCL|NaN|Inf|Killed|No space left" \
  "$RUN_ROOT/train.log"
```

检查 checkpoint：

```bash
find "$RUN_ROOT/checkpoints" -maxdepth 2 -type f -printf "%s %p\n" | sort -n
```

健康训练应满足：

- `NGPU` 个 rank 都存活；
- 所有可见 GPU 都有稳定显存和计算利用率；
- step 持续增加；
- loss、grad norm 和 learning rate 为有限值；
- checkpoint 到达目标步数后完整落盘；
- 磁盘空间没有快速耗尽。

### 9.6 正常结束审计

```bash
cat "$RUN_ROOT/exit_code"
tail -n 50 "$RUN_ROOT/train.log"
```

必须满足：

```text
exit_code = 0
train_exit_code = 0
checkpoint_delta_vs_base/watcher_exit_code = 0
日志到达 20000/20000
TRAIN_DONE rc=0
watch_status.json 中 completed_steps = [5000,10000,15000,20000]
```

检查恰好 4 个 checkpoint：

```bash
for step in 5000 10000 15000 20000; do
  ckpt="$RUN_ROOT/checkpoints/checkpoint_step_${step}"
  test -d "$ckpt"
  test -s "$ckpt/transformer/diffusion_pytorch_model.safetensors"
  test -s "$ckpt/transformer/config.json"
  du -sh "$ckpt"
done
```

如果 checkpoint 正在写入，不要在模型大文件还增长时复制、评测或计算最终 hash。

---

## 10. 训练 checkpoint 与基础模型的变化检查

这类检查与 RoboTwin 成功率评测不是同一件事。仓库提供独立程序 [`script/compare_checkpoint_to_base.py`](script/compare_checkpoint_to_base.py)，它不修改训练主循环，也不占用训练 GPU；默认在 CPU 上按 tensor chunk 流式读取 safetensors。

每个 checkpoint 保存完成后至少记录：

1. checkpoint 文件大小；
2. checkpoint 权重 SHA256；
3. 是否存在 NaN/Inf；
4. 与 `lingbot-va-base` 对应 transformer 权重的参数差分；
5. L2 范数、相对 L2、最大绝对差、参数余弦相似度；
6. 配置文件是否仍可被评测加载。

### 10.1 自动 watcher

如果按第 9.3 节使用：

```bash
bash script/run_robotwin_clean_zipbaseline_global64_20k_with_delta_audit.sh
```

则无需再手动启动 watcher。检查：

```bash
cat "$RUN_ROOT/checkpoint_delta_vs_base/watch_status.json"
tail -n 100 "$RUN_ROOT/checkpoint_delta_vs_base/watcher.log"
```

每个 checkpoint 会输出：

```text
checkpoint_step_5000_vs_base.json
checkpoint_step_5000_vs_base.txt
...
checkpoint_step_20000_vs_base.json
checkpoint_step_20000_vs_base.txt
```

`watch_status.json` 最终应为：

```json
{
  "completed_steps": [5000, 10000, 15000, 20000],
  "pending_steps": [],
  "failures": {}
}
```

### 10.2 手动比较单个 checkpoint

```bash
python script/compare_checkpoint_to_base.py compare \
  --base "$LINGBOT_ROOT/models/lingbot-va-base" \
  --checkpoint "$RUN_ROOT/checkpoints/checkpoint_step_5000" \
  --output "$RUN_ROOT/checkpoint_delta_vs_base/checkpoint_step_5000_vs_base.json"
```

程序同时生成同名 `.txt` 摘要。退出码为 0 且 `audit_ok=true` 才表示：

- base 和 checkpoint tensor key 完全匹配；
- 没有缺失或意外 tensor；
- base 和 checkpoint 都没有 NaN/Inf；
- 所有统计完成。

### 10.3 手动启动 watcher

如果训练是通过不带 watcher 的 portable 入口启动，可另开 shell：

```bash
mkdir -p "$RUN_ROOT/checkpoint_delta_vs_base"

nohup "$LINGBOT_ROOT/code/.venv/bin/python" \
  script/compare_checkpoint_to_base.py watch \
  --base "$LINGBOT_ROOT/models/lingbot-va-base" \
  --checkpoint-root "$RUN_ROOT/checkpoints" \
  --steps 5000,10000,15000,20000 \
  --output "$RUN_ROOT/checkpoint_delta_vs_base" \
  --poll-seconds 60 \
  --stable-polls 2 \
  > "$RUN_ROOT/checkpoint_delta_vs_base/watcher.log" 2>&1 &
```

### 10.4 指标解释

JSON 的 `overall` 包含：

| 字段 | 含义 |
|---|---|
| `changed_fraction` | 与 base 数值不完全相同的参数元素比例 |
| `delta_l2` | 所有参数差分的整体 L2 |
| `relative_l2_delta` | `delta_l2 / base_l2` |
| `cosine_similarity` | base 与 checkpoint 参数向量余弦相似度 |
| `rms_delta` | 参数差分 RMS |
| `mean_abs_delta` | 平均绝对差 |
| `max_abs_delta` | 最大绝对差 |
| `checkpoint_nonfinite` | checkpoint 中 NaN/Inf 元素数，必须为 0 |

还会按 module group 汇总，并列出 relative/absolute delta 最大的 tensor，便于判断变化集中在哪些层。

最低限度文件身份审计：

```bash
BASE="$LINGBOT_ROOT/models/lingbot-va-base/transformer/diffusion_pytorch_model.safetensors"

for step in 5000 10000 15000 20000; do
  CKPT="$RUN_ROOT/checkpoints/checkpoint_step_${step}/transformer/diffusion_pytorch_model.safetensors"
  test -s "$CKPT"
  sha256sum "$CKPT"
  du -h "$CKPT"
done
```

注意：

- watcher 使用 CPU，但会读取大模型文件；共享存储较慢时会产生额外 I/O；
- 不要同时手动启动多个 watcher 比较同一 checkpoint；
- 不要在 GPU 上加载完整 base 和 checkpoint；程序默认 CPU；
- 参数差异不直接等于策略成功率；
- “变化很小”不代表没训练，“变化很大”也不代表更好；
- 最终模型选择仍以第 11 节的固定 seed RoboTwin 成功率为准。

---

## 11. 对齐评测 5K/10K/15K/20K

### 11.1 前置条件

必须同时满足：

```text
训练已经停止
官方 50x20 校准已通过
LATEST_PASSED 指向有效 summary.json
目标 checkpoint 已完整写入
没有其他 RoboTwin 评测进程
```

检查：

```bash
cat "$LINGBOT_ROOT/train_out/robotwin-clean-calibration/LATEST_PASSED"
pgrep -af "run_robotwin_eval|run_server_ckpt|eval_polict_client_openpi" || true
nvidia-smi
```

### 11.2 评测单个 checkpoint

```bash
export LINGBOT_ROOT=/path/to/Lingbot-va
export NGPU=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

export CHECKPOINT_PATH="$RUN_ROOT/checkpoints/checkpoint_step_5000"
bash script/run_robotwin_clean_checkpoint_eval.sh
```

脚本会：

1. 读取最近一次通过的官方校准 summary；
2. 拒绝低于 85% 或非 50x20 的校准；
3. 使用固定 50-task seed；
4. 准备 checkpoint 的完整评测模型；
5. 运行 50 tasks x 20 episodes；
6. 生成并严格审计 `summary.json`。

也可显式指定校准：

```bash
export CALIBRATION_SUMMARY=/path/to/official-calibration/summary.json
export CHECKPOINT_PATH="$RUN_ROOT/checkpoints/checkpoint_step_10000"
bash script/run_robotwin_clean_checkpoint_eval.sh
```

### 11.3 串行评测全部 checkpoint

不要并行评测多个 checkpoint。串行运行：

```bash
for step in 5000 10000 15000 20000; do
  export CHECKPOINT_PATH="$RUN_ROOT/checkpoints/checkpoint_step_${step}"
  export RESULT_LABEL="checkpoint_step_${step}"
  NGPU=4 bash script/run_robotwin_clean_checkpoint_eval.sh
done
```

每轮使用独立 RUN_ID 和结果目录，不能把不同 checkpoint 的结果写进同一个目录。

### 11.4 summary.json 应包含

- 50 个任务；
- 每任务 20 episodes；
- 每任务 success / total / SR；
- 总 success、total 和 micro-average SR；
- mean / median / P95 episode timing；
- timing episode `1..20` 连续；
- seed 无重复、顺序正确且来自固定 cache；
- `res` total/success 与 timing 一致；
- `AUDIT_OK`。

总成功率：

$$
\mathrm{SR}
=
\frac{\sum_{t=1}^{50}\mathrm{success}_t}
{\sum_{t=1}^{50}\mathrm{episodes}_t}.
$$

完整评测定义、Easy/Hard 区别和断点恢复规则见 [`RoboTwin_Evaluation_Usage.md`](RoboTwin_Evaluation_Usage.md)。

---

## 12. 结果整理与汇报

每个 checkpoint 至少汇报：

| 字段 | 内容 |
|---|---|
| Model | checkpoint step 和完整路径 |
| Train data | Clean 50 tasks / 2500 trajectories |
| Steps | optimizer steps |
| Global batch | batch/GPU x GPU x GA |
| Scheduler | constant |
| Official calibration SR | 同机官方模型 50x20 SR |
| Checkpoint SR | 50x20 micro-average |
| Per-task SR | 50 行 |
| Wall time | 完整评测墙钟 |
| Throughput | episodes/hour |
| Timing | mean / median / P95 |
| Audit | `AUDIT_OK` |
| Recovery | 是否断点恢复、次数及原因 |

推荐表：

```text
official model
checkpoint_step_5000
checkpoint_step_10000
checkpoint_step_15000
checkpoint_step_20000
```

不要：

- 混合不同 seed；
- 混合每任务 5/10/20 episodes；
- 混合 Easy 与 Hard；
- 把临时进度 SR 当最终 SR；
- 忽略失败 episode；
- 只汇报总 SR 而不保留逐任务结果；
- 用官方模型在另一台未校准机器上的结果替代当前校准。

---

## 13. 断点恢复与安全规则

### 13.1 训练异常

当前 portable 训练脚本不承诺 optimizer state 的自动续训。出现异常时：

1. 保存完整日志和现场；
2. 记录最后完整 checkpoint；
3. 不覆盖旧 RUN_ID；
4. 判断代码是否支持从该 checkpoint 加载完整训练状态；
5. 不确定时启动新的 RUN_ID，并在报告中说明。

不要通过重命名 checkpoint 伪造 step。

### 13.2 评测异常

评测支持 partial resume，但规则固定：

1. 沿用同一个 RUN_ID；
2. 保留原结果目录；
3. `SKIP_EXISTING=1`；
4. 按 timing 中真实完成的 seed 继续；
5. 不覆盖完成 episode；
6. 恢复后 seed 仍必须有序且无重复；
7. 最终重新运行严格审计。

### 13.3 GPU holder

如果机器使用 GPU holder/guard，只能使用该机器提供的 guard 命令，不要直接 `kill` holder 或 daemon。通用原则与示例见 [`RoboTwin_Evaluation_Usage.md`](RoboTwin_Evaluation_Usage.md#10-gpu-holderguard)。

---

## 14. 常见故障

### 14.1 `No visible GPU found`

```bash
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
python -c "import torch; print(torch.cuda.device_count())"
```

确保 `NGPU` 与 `CUDA_VISIBLE_DEVICES` 的数量一致。

### 14.2 global batch 不整除

脚本会拒绝：

```text
TARGET_GLOBAL_BATCH is not divisible by batch_size*NGPU
```

使用 1/2/4/8/16/32/64 张 GPU，脚本会分别设置 64/32/16/8/4/2/1 的 accumulation，并始终保持 global batch 64。不要用 3、6、10 等不能整除 64 的 GPU 数启动这条正式协议；应调整可见 GPU 数，而不是改 global batch。

### 14.3 训练 attention mode 错

报错会明确指出基础模型不是 `flex`。重新执行第 5.4 节，只修改 `lingbot-va-base`。

### 14.4 官方评测 attention mode 错

官方模型必须是 `torch` 或 `flashattn`。不要把基础模型的修改命令应用到官方 posttrain 模型。

### 14.5 FlashAttention 编译失败

检查：

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
gcc --version
echo "$CUDA_HOME"
```

清理当前机器生成的失败 build cache 后重新编译，不要复制其他服务器的 wheel 或 `.so`，除非 CUDA、PyTorch、Python、GPU 架构完全一致并有可复现构建记录。

### 14.6 OOM

本基线保持 `batch_size=1` 和 activation checkpointing。先检查是否有其他进程占显存：

```bash
nvidia-smi
```

不要为了绕过 OOM 悄悄改变 action/video 配置、global batch 或模型精度。任何协议变化必须成为新实验。

### 14.7 `Vulkan DeviceLost` 或 RT 无进展

```bash
vulkaninfo --summary
rg -n "DeviceLost|Vulkan|svulkan|OIDN|timeout|no.progress" \
  "$LINGBOT_ROOT/logs/robotwin-clean-eval"
```

OIDN unsupported device/invalid handle 警告只有在 episode 仍持续推进并正常落盘时才可视为非致命；若步数停止或没有结果，必须视为异常。

### 14.8 官方模型 SR 低

不要归因于自有训练。回到第 8.6 节，先修复评测链路。

### 14.9 磁盘满

```bash
df -h "$LINGBOT_ROOT"
du -sh "$LINGBOT_ROOT"/* | sort -h
```

不要删除正在写入的 checkpoint、当前 RUN_ID 结果或固定 seed。可以清理明确可重建的下载 cache，但应先确认模型和数据本体不是 cache 内的软链接。

### 14.10 SwanLab 无法联网

切换离线模式不会改变训练数学协议：

```bash
export LINGBOT_SWANLAB_MODE=offline
```

本地日志和 checkpoint 仍必须保存。

---

## 15. 一页式执行清单

以下是从空服务器到完成实验的顺序，不能跳过第 8 步：

```text
1. 克隆 seamoke/WAM-JUDGE 到 $LINGBOT_ROOT/code
2. 安装 Ubuntu 构建、Vulkan、FFmpeg 依赖
3. 创建 Python 3.10 venv
4. 安装 PyTorch 2.9.0 cu126 和固定 Python 依赖
5. 下载 lingbot-va-base
6. 下载 lingbot-va-posttrain-robotwin
7. 下载完整 robotwin-clean-and-aug-lerobot
8. 审计 Clean 50 tasks / 2500 episodes / latent
9. 安装 RoboTwin、cuRobo、PyTorch3D 和仿真资产
10. 运行官方模型 50x20 校准，必须 SR >= 85%
11. 运行 Clean-only ZIP baseline 20K 训练，global batch 固定 64
12. 核验 5K/10K/15K/20K 四个 checkpoint
13. 训练停止后串行评测四个 checkpoint
14. 审计每轮 summary.json
15. 汇总逐任务 SR、总 SR、墙钟、吞吐和 timing
```

最终需要保存：

```text
代码 commit 和 git diff
environment-freeze.txt
nvidia-smi
Clean dataset audit JSON
官方模型 calibration summary.json
训练 run_manifest.txt 和 train.log
四个 checkpoint
四轮 checkpoint evaluation summary.json
异常和恢复记录
```

完成这些步骤后，实验才具备可复现、可比较和可汇报的证据链。
