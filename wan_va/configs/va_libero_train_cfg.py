# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg
import os

va_libero_train_cfg = EasyDict(__name__='Config: VA libero train')
va_libero_train_cfg.update(va_libero_cfg)

va_libero_train_cfg.dataset_path = '/workspace/lingbot-va/data/libero-long-lerobot'
va_libero_train_cfg.empty_emb_path = os.path.join(va_libero_train_cfg.dataset_path, 'empty_emb.pt')
va_libero_train_cfg.enable_swanlab = True
va_libero_train_cfg.save_root = '/workspace/lingbot-va/train_out/libero'

# --- Speed-optimized (4x H200, plenty of VRAM) ---
# batch_size=1: no variable-F padding; shorter flex-attn sequence per forward.
# global_batch = batch_size * NGPU * gradient_accumulation_steps = 1 * 4 * 4 = 16
va_libero_train_cfg.batch_size = 1
va_libero_train_cfg.gradient_accumulation_steps = 4
# Off saves ~1.5-2x step time; turn back on if OOM.
va_libero_train_cfg.enable_activation_checkpointing = False
va_libero_train_cfg.load_worker = 4
va_libero_train_cfg.gc_interval = 300
va_libero_train_cfg.save_interval = 1000
va_libero_train_cfg.cfg_prob = 0.1

# Training parameters
va_libero_train_cfg.learning_rate = 1e-5
va_libero_train_cfg.beta1 = 0.9
va_libero_train_cfg.beta2 = 0.95
va_libero_train_cfg.weight_decay = 1e-1
va_libero_train_cfg.warmup_steps = 10
# Original recipe: 5000 steps @ global_batch=40 -> 200k clip exposures.
# This config: global_batch=16 -> 12500 steps for the same 200k exposures.
va_libero_train_cfg.num_steps = 12000
