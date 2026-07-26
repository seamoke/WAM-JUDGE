# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

va_robotwin_train_cfg.dataset_path = os.environ.get(
    'ROBOTWIN_DATASET_PATH',
    '/data/lingbot-va/models/datasets/robotwins-short/robotwin-clean-and-aug-lerobot',
)
va_robotwin_train_cfg.empty_emb_path = os.environ.get(
    'ROBOTWIN_EMPTY_EMB_PATH',
    os.path.join(va_robotwin_train_cfg.dataset_path, 'empty_emb.pt'),
)
va_robotwin_train_cfg.enable_swanlab = _env_bool('LINGBOT_ENABLE_SWANLAB', True)
va_robotwin_train_cfg.save_root = os.environ.get(
    'LINGBOT_TRAIN_SAVE_ROOT', '/workspace/lingbot-va/train_out/robotwin'
)

# Skip segments longer than this (end_frame - start_frame); reduces OOM and speeds up training.
va_robotwin_train_cfg.max_episode_frames = int(
    os.environ.get('LINGBOT_MAX_EPISODE_FRAMES', '500')
)

# global_batch = batch_size * world_size * gradient_accumulation_steps
va_robotwin_train_cfg.batch_size = int(os.environ.get('LINGBOT_TRAIN_BATCH_SIZE', '1'))
va_robotwin_train_cfg.gradient_accumulation_steps = int(
    os.environ.get('LINGBOT_GRADIENT_ACCUMULATION_STEPS', '16')
)
va_robotwin_train_cfg.enable_activation_checkpointing = _env_bool(
    'LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING', False
)
va_robotwin_train_cfg.dataset_init_workers = int(
    os.environ.get('LINGBOT_DATASET_INIT_WORKERS', '128')
)
va_robotwin_train_cfg.load_worker = int(os.environ.get('LINGBOT_TRAIN_LOAD_WORKERS', '2'))
va_robotwin_train_cfg.gc_interval = int(os.environ.get('LINGBOT_GC_INTERVAL', '1000'))
va_robotwin_train_cfg.save_interval = int(os.environ.get('LINGBOT_SAVE_INTERVAL', '3000'))
va_robotwin_train_cfg.save_steps = [
    int(step)
    for step in os.environ.get('LINGBOT_SAVE_STEPS', '').split(',')
    if step.strip()
]
va_robotwin_train_cfg.cfg_prob = float(os.environ.get('LINGBOT_CFG_PROB', '0.1'))

# Training parameters
va_robotwin_train_cfg.learning_rate = 1e-5
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = int(os.environ.get('LINGBOT_WARMUP_STEPS', '1000'))
va_robotwin_train_cfg.num_steps = int(os.environ.get('LINGBOT_TRAIN_NUM_STEPS', '21000'))
va_robotwin_train_cfg.lr_scheduler_type = os.environ.get(
    'LINGBOT_LR_SCHEDULER', 'constant'
)
va_robotwin_train_cfg.min_lr_ratio = float(
    os.environ.get('LINGBOT_MIN_LR_RATIO', '0.0')
)
