# Part 1: Base SFT + VLAC

Part 1 包含固定数据、Stage 1 Base SFT 和 VLAC 全参数训练。Part 2（RFT）尚未完成。

## 1. 固定设置

每个 RoboTwin task 使用同一份 50 Clean + 50 Randomized：

| 数据 | Stage 1 | Stage 2 | 合计 |
|---|---:|---:|---:|
| Clean | 30 | 20 | 50 |
| Randomized | 30 | 20 | 50 |

- Base SFT：只用 Stage 1，共 3000 条轨迹。
- VLAC：使用 Stage 1 + Stage 2，共 5000 条轨迹。
- Base：15,000 optimizer steps，global batch 64。
- Base checkpoint：3K、6K、9K、12K、15K。
- VLAC-2B：全参数训练，不使用 LoRA，不冻结 ViT/aligner。

```bash
export LINGBOT_ROOT=/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va
export PROJECT_ROOT="$LINGBOT_ROOT/code"
export SOURCE_DATA_ROOT="$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot"
export PREPARED_DATA_ROOT="$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42"
cd "$PROJECT_ROOT"
```

每次训练前检查 GPU：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

存在未知 GPU 进程时不要启动。

## 2. 一次性准备数据

只在首次执行：

```bash
source "$PROJECT_ROOT/.venv/bin/activate"

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

必须看到：

```text
TWO_STAGE_DATASET_PREPARATION_OK
```

以后只校验，不重新划分：

```bash
python script/prepare_robotwin_two_stage_dataset.py \
  --output-root "$PREPARED_DATA_ROOT" \
  --allow-missing-latent-segments 8 \
  --verify-only
```

不要修改或删除：

```text
$PREPARED_DATA_ROOT/split_manifest.json
$PREPARED_DATA_ROOT/PREPARATION_COMPLETE.json
```

## 3. 训练 Stage 1 Base

```bash
export RUN_ID="robotwin_stage1_4xh100_$(date +%Y%m%d_%H%M%S)"
bash script/run_robotwin_stage1_sft_portable.sh
```

输出：

```text
$LINGBOT_ROOT/train_out/robotwin/$RUN_ID/
```

训练结束后检查：

```bash
for step in 3000 6000 9000 12000 15000; do
  test -d "$LINGBOT_ROOT/train_out/robotwin/$RUN_ID/checkpoints/checkpoint_step_$step"
done
```

日志必须到达 `15000/15000`。

## 4. 准备 VLAC

脚本会校验固定 manifest、创建独立环境、下载 `InternRobotics/VLAC`、构造 RGB pair
并校验清单。

先准备 2-task smoke：

```bash
bash script/robotwin_two_stage_vlac_prepare.sh smoke
```

再准备全部 50 tasks：

```bash
bash script/robotwin_two_stage_vlac_prepare.sh full
```

输出：

```text
$LINGBOT_ROOT/train_out/critic/robotwin/models/VLAC-2B
$LINGBOT_ROOT/train_out/critic/robotwin/vlac_finetune/two_stage_smoke
$LINGBOT_ROOT/train_out/critic/robotwin/vlac_finetune/two_stage_full
```

`full/val.jsonl` 只用于监控，因为固定 50+50 的全部 pair 都参与优化。正式指标必须
使用未参与训练的 simulator seed、场景或 task split。

## 5. 训练 VLAC

先跑四卡 10-step smoke：

```bash
bash script/robotwin_two_stage_vlac_train.sh smoke
```

必须满足：

```text
10/10 optimizer steps
checkpoint-10 exists
numeric_output_ok: true
voc_finite: true
vroc_finite: true
smoke_passed: true
```

smoke 通过后启动 full：

```bash
bash script/robotwin_two_stage_vlac_train.sh full
```

10-step 只验证训练和评测链路，不能作为最终 critic 结果。正式结果报告独立测试集的
sign accuracy、macro-F1、AUC、Spearman、VOC/VROC 和 antisymmetry MAE。

## 6. 上传 Base 12K/15K Checkpoint

Base 训练完成后，负责人必须把以下目录上传到需求方指定的 ModelScope 私有仓库：

```text
checkpoint_step_12000
checkpoint_step_15000
```

不要把 token 写入代码、文档、Git 或日志。安装工具：

```bash
python -m pip install -U modelscope-hub
```

先在 ModelScope 网页创建私有模型仓库，然后执行：

```bash
read -s -p "ModelScope token: " MODELSCOPE_API_TOKEN
echo
export MODELSCOPE_API_TOKEN
export MODELSCOPE_REPO_ID="YOUR_NAMESPACE/lingbot-va-stage1-checkpoints"
export STAGE1_RUN_DIR="$LINGBOT_ROOT/train_out/robotwin/$RUN_ID"

bash script/upload_robotwin_stage1_checkpoints_modelscope.sh

unset MODELSCOPE_API_TOKEN
```

上传完成后，把 ModelScope 仓库链接发给需求方。

参考：https://www.modelscope.cn/docs/models/upload

## 7. 完成检查

- Base 到达 15K；
- 12K/15K checkpoint 已上传并发送链接；
- VLAC smoke 通过；
- VLAC full 完成；
- 独立测试集指标已保存；
- GPU 无残留进程。
