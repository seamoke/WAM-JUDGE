# RoboTwin 新服务器统一评测注意事项

## 0. Clean-only 正式入口

本代码快照只把 Easy/Clean 作为当前训练与 checkpoint 对比主线。推荐入口：

```bash
# 第一步：官方模型 50 tasks x 20 episodes 校准
NGPU=8 bash script/run_robotwin_clean_official_calibration.sh

# 第二步：只有官方 SR >= 85% 后才允许评测自有 checkpoint
CHECKPOINT_PATH=/path/to/checkpoint_step_5000 \
NGPU=8 \
bash script/run_robotwin_clean_checkpoint_eval.sh
```

两条脚本都会调用严格审计器 `script/audit_robotwin_clean_eval.py`。官方模型未达到 85% 时，应视为环境、渲染、模型配置或评测协议未对齐，不能用自有 checkpoint 的结果判断训练质量。

## 1. 目标与原则

新服务器可以产生和使用自己的新 checkpoint，不需要复制或对齐旧服务器上的
`checkpoint_step_18000`、`checkpoint_step_21000` 等权重。

需要统一的是：

- 代码版本与必要修复；
- RoboTwin 数据、任务定义和场景资产；
- Python、CUDA、PyTorch、SAPIEN、svulkan2 等运行环境；
- Easy/Hard 的种子缓存；
- 推理、渲染和模型输入协议；
- episode 数量、成功率计算方式和结果审计规则。

硬件和 checkpoint 可以不同，但每次结果必须记录实际硬件、checkpoint 路径、代码版本和协议。只有协议一致的实验才能直接比较成功率。

## 2. 推荐目录结构

建议在新服务器保留统一结构：

```text
/workspace/lingbot-va/
├── .venv/
├── models/
│   └── lingbot-va-posttrain-robotwin/
├── train_out/
│   ├── robotwin-short-3gpu/checkpoints/
│   │   └── checkpoint_step_XXXXX/
│   └── robotwin/eval_seed_cache/
│       ├── demo_clean_seed0_n100.json
│       └── demo_randomized_seed0_n100.json
├── evaluation/
├── logs/
└── scripts/
```

新 checkpoint 只要放入约定目录并在 RUN_ID 和日志中明确记录即可。禁止通过重命名其他权重来伪造 checkpoint。

## 3. 新服务器环境准备

建议创建独立虚拟环境：

```bash
cd /workspace/lingbot-va
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

当前参考环境的核心信息：

```text
Python environment: /workspace/lingbot-va/.venv
PyTorch: 2.9.0+cu128
CUDA: 12.8 系列
```

新服务器不必机械复制所有版本，但正式对比前必须验证并保存：

```bash
python --version
pip freeze
nvidia-smi
vulkaninfo --summary
```

重点检查：

- PyTorch 能识别全部目标 GPU；
- CUDA 推理正常；
- Vulkan RT 能正常创建设备和渲染；
- OptiX denoiser 可用；
- SAPIEN、svulkan2、OpenCV、FFmpeg 等依赖可用；
- 模型不会因为显存配置而意外切换精度或 offload 策略。

建议把以下信息保存到每轮实验日志：

```bash
git rev-parse HEAD
git status --short
python --version
pip freeze
nvidia-smi
```

## 4. 必须保留的评测修复

迁移代码时，应包含当前已经验证过的修复：

1. Partial evaluation 恢复时，依据 timing 中实际完成的 seed 继续。
2. 使用 seed cache 时，如果缺少可靠 timing，拒绝盲目恢复。
3. 恢复后的 seed 必须无重复、无乱序。
4. `place_object_scale` 在首次 `check_success()` 前完成 `arm_tag` 初始化。
5. 动态任务队列使用文件锁，避免多个 worker 重复领取任务。
6. 使用 `SKIP_EXISTING=1` 保留已完成结果。
7. 每个 episode 独立写入 timing，支持安全断点续跑。
8. 异常恢复必须沿用同一个 RUN_ID，不覆盖旧结果。

不要只复制单个启动脚本。应同步完整评测代码、未提交补丁、任务代码和审计脚本。

## 5. 模型与 checkpoint 记录

每次评测必须明确记录：

```text
Base model 路径
Checkpoint 路径
Checkpoint step
模型文件 SHA256
代码 commit/diff
训练配置摘要
```

例如：

```bash
sha256sum /workspace/lingbot-va/train_out/robotwin-short-3gpu/checkpoints/checkpoint_step_XXXXX/transformer/*
```

如果比较多个新 checkpoint：

- 所有 checkpoint 使用同一评测代码；
- 使用同一 Easy/Hard seed；
- 使用相同 episode 数；
- 使用相同推理、相机和渲染设置；
- 每个 checkpoint 使用独立 RUN_ID 和结果目录；
- 不把不同 episode 数的结果放在同一列直接比较。

## 6. Easy 与 Hard 数据定义

两个评测集必须分开运行和汇报：

| 名称 | task_config | 含义 |
|---|---|---|
| Easy/Clean | `demo_clean` | 标准、清洁场景 |
| Hard/Aug | `demo_randomized` | 光照、材质、背景、物体等随机化场景 |

二者使用同一套 50 个任务，但初始状态和视觉随机化不同。禁止将 Easy 与 Hard 的 episode 混合计算成一个成功率。

## 7. 固定种子

推荐继续使用已预采样并审计的缓存：

```text
Easy:
demo_clean_seed0_n100.json
SHA256=4794de71cd3140d2a415c72e9a246a5c7373080d3ef72b430a873930d028926c

Hard:
demo_randomized_seed0_n100.json
SHA256=4704731001e5890b677fb5f4f8d7da8e54502ba156d25c09812f677e5305139f
```

缓存中每任务可以包含 100 个候选 seed，但正式评测不必运行 100 次：

- 快速 checkpoint 对比：每任务 10 episodes；
- 最终正式汇报：建议每任务 20 episodes；
- 所有待比较 checkpoint 必须取同一缓存中的相同顺序 seed。

禁止为每个 checkpoint 重新随机采样，否则失去配对比较条件。

## 8. 固定评测协议

推荐固定参数：

```text
50 tasks
CLIENTS_PER_GPU=1
EXPERT_CHECK=1
FAST=0
LOW_RENDER=0
POLICY_CAMERAS_ONLY=0
DEFER_RENDER_UPDATES=0
RECREATE_CAMERAS_EVERY=0
RT_DENOISER=optix
WAN_VA_ENABLE_OFFLOAD=0
WAN_VA_OFFLOAD_VAE=0
WAN_VA_OFFLOAD_TEXT_ENCODER=1
timing=1
resume_partial=1
client_episode_chunk=1
timeout=900
no_progress_retries=3
SKIP_EXISTING=1
视频保存关闭
可视化关闭
prompt cache strict=1
```

特别注意：

- `RECREATE_CAMERAS_EVERY` 必须为 0；
- 不得用 `FAST=1`、低渲染或减少策略相机输入伪装正式结果；
- `DEFER_RENDER_UPDATES` 默认保持 0；
- 正式对比中不得临时改变 offload 策略；
- RT denoiser 必须保持一致，不能一部分使用 OptiX、一部分关闭。

## 9. GPU 与并行配置

并行度可以按新服务器 GPU 数量调整，但单 episode 协议不能变化。

六张 RTX 5090 的参考配置：

```text
GPU 0-4: 五个动态评测 shard
GPU 5: prompt embedding service
NGPU=5
CLIENTS_PER_GPU=1
```

报告中必须记录：

- GPU 型号和数量；
- 每张 GPU 的角色；
- shard 数量；
- prompt embedding service 是否独占 GPU；
- 总墙钟时间；
- episode 吞吐；
- 单 episode mean、median、P95 timing。

RTX 5090 与 H200 的速度可单独比较，但必须先确认渲染一致。H200 曾出现 RT/Vulkan 兼容问题，不能因为 CUDA 推理成功就认定完整评测环境正常。

## 10. GPU holder/guard

如果服务器使用 GPU holder 保活：

- 只能通过现有 guard 命令释放和恢复；
- 禁止直接 `kill` 或 `pkill` holder/daemon；
- 单次 release 最长 60 分钟，禁止释放 120 分钟；
- 正式任务仍活跃时，不要强行启动 holder 与任务争抢显存；
- 任务结束且确认无评测残留后再恢复 holder；
- 恢复后检查 holder 数量和显存占用是否稳定。

示例：

```bash
/root/bin/gpu-guard release --minutes 60
/root/bin/gpu-guard status
/root/bin/gpu-guard hold-now
```

长时间评测需要在不超过 60 分钟的限制下续期，不能一次设置超长 release。

## 11. 新服务器 smoke test

正式评测前，至少运行以下代表性任务：

```text
click_bell
move_stapler_pad
place_shoe
place_object_stand
place_object_scale
```

每项运行 1 至 3 个 episode。必须确认：

- 两类数据集都能加载；
- RT 画面正常；
- episode 步数持续推进；
- 推理 server/client 通信健康；
- `res` 和 `timing` 正常落盘；
- 没有 Vulkan DeviceLost、OOM 或未处理 AttributeError；
- `move_stapler_pad` 不会长时间无进展；
- `place_object_scale` 不再出现 `arm_tag` 初始化错误；
- seed 与缓存顺序一致。

端口探测产生的 opening handshake traceback 可以忽略，但真实客户端通信失败不能忽略。

## 12. RUN_ID 与断点恢复

每个实验使用唯一 RUN_ID，例如：

```text
checkpointXXXXX_easy_n10_5shards_6x5090_YYYYMMDD_HHMM
checkpointXXXXX_hard_n10_5shards_6x5090_YYYYMMDD_HHMM
```

RUN_ID 中建议包含：

- checkpoint step；
- Easy/Hard；
- 每任务 episode 数；
- shard/GPU 配置；
- 时间戳。

断点恢复规则：

1. 保留原 RUN_ID。
2. 保留原结果目录。
3. 使用 `SKIP_EXISTING=1`。
4. 根据 timing 中真实 seed 序列恢复。
5. 不覆盖已完成任务。
6. 不删除旧结果后“重新开始”来掩盖异常。

## 13. 完成后的严格审计

每个 Easy/Hard RUN_ID 都要检查：

1. 恰好 50 个任务。
2. 每任务恰好 10 或 20 episodes。
3. timing episode 连续为 `1..N`。
4. seed 无重复、无乱序。
5. seed 是该任务缓存的严格有序子序列。
6. `res.total` 等于 timing 条目数。
7. `res.success` 等于 timing 中 success 数量。
8. 没有未处理 OOM、DeviceLost、timeout 或 AttributeError。
9. 每任务成功率和总成功率可以从原始结果重新计算。

总成功率使用 episode micro-average：

$$
\mathrm{SR} =
\frac{\sum_{t=1}^{50}\mathrm{success}_t}
{\sum_{t=1}^{50}\mathrm{episodes}_t}
$$

当每个任务 episode 数相同，它也等于 50 个任务成功率的算术平均值。

## 14. 每轮实验应输出的报告

建议生成一个固定格式的 Markdown/JSON 报告，包含：

- 服务器和 GPU 信息；
- 代码 commit 与 diff 状态；
- Python/CUDA/PyTorch/Vulkan 环境；
- Base model 与 checkpoint 路径、step、SHA256；
- Easy/Hard seed cache SHA256；
- 完整协议参数；
- 50 个任务各自的成功数、总数和 SR；
- 总 SR；
- 总墙钟时间和吞吐；
- episode timing 的 mean、median、P95；
- 异常、恢复和跳过记录；
- 最终审计是否通过。

## 15. 启动正式评测前的最终检查

```text
[ ] 代码和评测修复已同步
[ ] RoboTwin 50任务及全部资产可访问
[ ] Base model 和新 checkpoint 可加载
[ ] checkpoint 路径及SHA已记录
[ ] Easy/Hard seed cache SHA正确
[ ] TEST_NUM已明确为10或20
[ ] RT/Vulkan/OptiX smoke通过
[ ] FAST/LOW_RENDER/RECREATE均保持正式设置
[ ] prompt embedding service健康
[ ] 动态shard无重复领任务
[ ] timing与partial resume已验证
[ ] RUN_ID和结果目录唯一
[ ] GPU guard使用方式正确
[ ] 完成后审计脚本和报告路径已准备
```

只要以上项目固定，新服务器可以自由产生新的 checkpoint，也可以使用不同数量或型号的 GPU。成功率比较依然具有一致的实验定义，硬件差异则单独体现在墙钟时间和吞吐中。
