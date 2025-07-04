# Copyright (c) 2025 Hansheng Chen

import numpy as np
import torch

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class GMFlowSDESchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class GMFlowSDEScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            shift: float = 1.0,
            output_mode='sample',
            variance_mode='large'):
        assert output_mode in ['sample', 'mean']
        assert variance_mode in ['large', 'small']

        sigmas = torch.from_numpy(1 - np.linspace(
            0, 1, num_train_timesteps, dtype=np.float32, endpoint=False))
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device=None):
        self.num_inference_steps = num_inference_steps

        sigmas = torch.from_numpy(1 - np.linspace(
            0, 1, num_inference_steps, dtype=np.float32, endpoint=False))
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        self.timesteps = (sigmas * self.config.num_train_timesteps).to(device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
            prediction_type='u',
            eps=1e-6) -> Union[GMFlowSDESchedulerOutput, Tuple]:
        assert prediction_type in ['u', 'x0']

        if isinstance(timestep, int) \
                or isinstance(timestep, torch.IntTensor) \
                or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    'Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to'
                    ' `EulerDiscreteScheduler.step()` is not supported. Make sure to pass'
                    ' one of the `scheduler.timesteps` as a timestep.'
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        ori_dtype = model_output.dtype
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        alpha = 1 - sigma
        alpha_to = 1 - sigma_to
        sigma_to_over_sigma = sigma_to / sigma.clamp(min=eps)
        alpha_over_alpha_to = alpha / alpha_to.clamp(min=eps)
        beta_over_sigma_sq = 1 - (sigma_to_over_sigma * alpha_over_alpha_to) ** 2

        if prediction_type == 'u':
            x0 = sample - sigma * model_output.to(torch.float32)
        else:
            x0 = model_output.to(torch.float32)
        noise = randn_tensor(
            model_output.shape, dtype=torch.float32, device=model_output.device, generator=generator)

        c1 = sigma_to_over_sigma ** 2 * alpha_over_alpha_to
        c2 = beta_over_sigma_sq * alpha_to
        if self.config.output_mode == 'sample' or self.config.variance_mode == 'small':
            std = beta_over_sigma_sq ** 0.5 * sigma_to  # c3_sqrt
        else:  # equivalent to ddpm large variance
            sqrt_beta = beta_over_sigma_sq ** 0.5 * sigma
            std = sqrt_beta * ((alpha_to ** 2 + sigma_to ** 2) / (alpha ** 2 + sigma ** 2).clamp(min=eps)) ** 0.5

        prev_sample = c1 * sample + c2 * x0 + std * noise

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(ori_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return GMFlowSDESchedulerOutput(prev_sample=prev_sample)
