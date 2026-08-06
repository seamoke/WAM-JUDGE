"""Batched first-chunk inference without modifying the upstream VA server."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

from wan_va.utils import data_seq_to_patch
from wan_va.wan_va_server import VA_Server


def collapse_cfg_batch(
    prediction: torch.Tensor, *, scale: float, enabled: bool
) -> torch.Tensor:
    """Map the shared 2B CFG transformer output back to model batch B."""
    if not enabled:
        return prediction
    if prediction.shape[0] % 2:
        raise ValueError(f"CFG prediction batch must be even: {prediction.shape}")
    positive, negative = prediction.chunk(2, dim=0)
    return negative + scale * (positive - negative) if scale > 1 else positive


class BatchedVAServer(VA_Server):
    """Extend the official first-chunk path to a configurable context batch."""

    def _reset_for_batch(self, prompts: Sequence[str] | None, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        super()._reset(prompt=None if prompts is None else list(prompts))
        self.transformer.clear_cache(self.cache_name)
        patch_size = self.job_config.patch_size
        latent_tokens = (
            self.job_config.frame_chunk_size
            * self.latent_height
            * self.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_tokens = (
            self.job_config.frame_chunk_size * self.job_config.action_per_frame
        )
        self.transformer.create_empty_cache(
            self.cache_name,
            self.job_config.attn_window,
            latent_tokens,
            action_tokens,
            dtype=self.dtype,
            device=self.device,
            batch_size=batch_size * (2 if self.use_cfg else 1),
        )

    def _repeat_input_for_cfg(self, input_dict):
        batch_size = int(input_dict["noisy_latents"].shape[0])
        if self.use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(
                2, 1, 1, 1, 1
            )
            input_dict["text_emb"] = torch.cat(
                [
                    self.prompt_embeds.to(self.dtype),
                    self.negative_prompt_embeds.to(self.dtype),
                ],
                dim=0,
            )
            repeats = 2 * batch_size
        else:
            repeats = batch_size
        input_dict["grid_id"] = input_dict["grid_id"][None].repeat(
            repeats, 1, 1
        )
        input_dict["timesteps"] = input_dict["timesteps"][None].repeat(
            repeats, 1
        )
        return input_dict

    def _encode_observation_batch(self, observations: Sequence[dict]) -> torch.Tensor:
        encode_one = super()._encode_obs
        encoded = []
        for observation in observations:
            # StreamingVAE caches represent temporal continuation. Each Q is an
            # independent trajectory, so sharing that cache across batch items
            # would incorrectly treat Q[n+1] as the next frame of Q[n].
            self.streaming_vae.clear_cache()
            if self.env_type == "robotwin_tshape":
                self.streaming_vae_half.clear_cache()
            encoded.append(encode_one(observation))
        self.streaming_vae.clear_cache()
        if self.env_type == "robotwin_tshape":
            self.streaming_vae_half.clear_cache()
        if any(value is None or value.shape[0] != 1 for value in encoded):
            raise ValueError("Every first-chunk observation must encode to batch one")
        return torch.cat(encoded, dim=0)

    def _seeded_noise(self, shape: tuple[int, ...], seeds: Sequence[int]) -> torch.Tensor:
        samples = []
        for seed in seeds:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            samples.append(
                torch.randn(
                    (1, *shape),
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        return torch.cat(samples, dim=0)

    def postprocess_action_batch(self, action: torch.Tensor) -> torch.Tensor:
        action = action.detach().cpu()[..., 0]
        if self.action_norm_method != "quantiles":
            raise NotImplementedError(self.action_norm_method)
        action = (action + 1) / 2 * (
            self.actions_q99[None] - self.actions_q01[None] + 1e-6
        ) + self.actions_q01[None]
        return action[:, self.job_config.used_action_channel_ids]

    def infer_batch(
        self,
        observations: Sequence[dict],
        prompts: Sequence[str],
        seeds: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(observations)
        if len(prompts) != batch_size or len(seeds) != batch_size:
            raise ValueError("observations, prompts, and seeds must have equal length")
        self._reset_for_batch(prompts, batch_size)
        init_latent = self._encode_observation_batch(observations)
        frame_chunk_size = int(self.job_config.frame_chunk_size)
        latents = self._seeded_noise(
            (
                48,
                frame_chunk_size,
                self.latent_height,
                self.latent_width,
            ),
            seeds,
        )
        actions = self._seeded_noise(
            (
                self.job_config.action_dim,
                frame_chunk_size,
                self.action_per_frame,
                1,
            ),
            [int(seed) + 10_000_019 for seed in seeds],
        )

        self.scheduler.set_timesteps(self.job_config.num_inference_steps)
        self.action_scheduler.set_timesteps(
            self.job_config.action_num_inference_steps
        )
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), value=0)
        if self.job_config.video_exec_step != -1:
            timesteps = timesteps[: self.job_config.video_exec_step]
        action_timesteps = F.pad(
            self.action_scheduler.timesteps, (0, 1), value=0
        )

        cfg_batch = batch_size * (2 if self.use_cfg else 1)
        with torch.no_grad():
            for index, timestep in enumerate(timesteps):
                last_step = index == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(self.dtype)
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    timestep,
                    timestep,
                    latent_cond,
                    None,
                    frame_st_id=0,
                )
                prediction = self.transformer(
                    self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )
                if not last_step or self.job_config.video_exec_step != -1:
                    prediction = data_seq_to_patch(
                        self.job_config.patch_size,
                        prediction,
                        frame_chunk_size,
                        self.latent_height,
                        self.latent_width,
                        batch_size=cfg_batch,
                    )
                    prediction = collapse_cfg_batch(
                        prediction,
                        scale=float(self.job_config.guidance_scale),
                        enabled=self.use_cfg,
                    )
                    latents = self.scheduler.step(
                        prediction, timestep, latents, return_dict=False
                    )
                latents[:, :, 0:1] = latent_cond

            for index, timestep in enumerate(action_timesteps):
                last_step = index == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    (
                        batch_size,
                        self.job_config.action_dim,
                        1,
                        self.action_per_frame,
                        1,
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )
                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    timestep,
                    timestep,
                    None,
                    action_cond,
                    frame_st_id=0,
                )
                prediction = self.transformer(
                    self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True,
                )
                if not last_step:
                    prediction = rearrange(
                        prediction,
                        "b (f n) c -> b c f n 1",
                        f=frame_chunk_size,
                    )
                    prediction = collapse_cfg_batch(
                        prediction,
                        scale=float(self.job_config.action_guidance_scale),
                        enabled=self.use_cfg,
                    )
                    actions = self.action_scheduler.step(
                        prediction, timestep, actions, return_dict=False
                    )
                actions[:, :, 0:1] = action_cond

        actions[:, ~self.action_mask] *= 0
        result = self.postprocess_action_batch(actions)
        torch.cuda.empty_cache()
        return result, latents
