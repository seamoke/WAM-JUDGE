# Faster LIBERO eval: fewer denoising steps (quality trade-off for throughput).
from easydict import EasyDict

from .va_libero_cfg import va_libero_cfg

va_libero_eval_cfg = EasyDict(__name__='Config: VA libero eval (fast)')
va_libero_eval_cfg.update(va_libero_cfg)

va_libero_eval_cfg.num_inference_steps = 10
va_libero_eval_cfg.action_num_inference_steps = 20
