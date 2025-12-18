


# math functions for sampling schedule
import math
from typing import Callable, Literal

import torch


class TimestepDistUtils:
    
    @staticmethod
    def t_shift(mu: float, sigma: float, t: torch.Tensor):
        """
            see eq.(12) of https://arxiv.org/abs/2506.15742 Black Forest Labs (2025)
            t' = \frac{e^{\mu}}{e^{\mu} + (1/t - 1)^{\sigma}}
        """
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    @staticmethod
    def lerp_mu( # qwen params
        seq_len,
        min_seq_len: int = 256,
        max_seq_len: int = 8192,
        min_mu: float = 0.5, 
        max_mu: float = 0.9,
        train_dist: str = "linear",
    ):
        """
        Resolution-dependent shifting of timestep schedules
        from Esser et al. https://arxiv.org/abs/2403.03206 
        updated with default params for Qwen
        """
        m = (max_mu - min_mu) / (max_seq_len - min_seq_len)
        b = min_mu - m * min_seq_len
        mu = seq_len * m + b
        return mu

    @staticmethod
    def logit_normal(t, mu=0.0, sigma=1.0):
        """
            Logit normal PDF, as in logistic(randn(mu, sigma))
        """
        pdf = torch.zeros_like(t)
        z = (torch.logit(t) - mu) / sigma
        coef = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
        pdf = coef * torch.exp(-0.5 * z**2) / (t * (1.0 - t))
        return pdf
    
    @staticmethod
    def scaled_clipped_gaussian(t):
        """
            Heuristic distribution for gaussian wuth mu = 0.5 and sigma=0.5, 
            clipped to [0,1], with int_0^1dt =1.0
        """
        y = torch.exp(-2 * (t - 0.5) ** 2)
        y = (y - 0.606) * 4.02
        return y
    
    @staticmethod
    def get_seq_len(latents):
        if latents.dim() == 4 or latents.dim() == 5:
            h,w = latents.shape[-2:]
            seq_len = (h//2)*(w//2)
        elif latents.dim() == 3:
            seq_len = latents.shape[1] # [B, L=h*w, C]
        else:
            raise ValueError(f"{latents.dim()=} not in 3,4,5")
        return seq_len

    def __init__(
        self,
        min_seq_len=256,
        max_seq_len=8192,
        min_mu=0.5,
        max_mu=0.9,
        train_dist:Literal["logit-normal", "linear"]="linear",
        train_shift:bool=True,
        inference_dist:Literal["logit-normal", "linear"]="linear",
        inference_shift:bool=True,
        static_mu:float|None=None,
        loss_weight_dist: Literal["scaled_clipped_gaussian", "logit-normal"] | None = None,
    ):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.train_dist = train_dist
        self.train_shift = train_shift
        self.inference_dist = inference_dist
        self.inference_shift = inference_shift
        self.static_mu = static_mu
        self.loss_weight_dist = loss_weight_dist

    def lin_t_to_dist(self, t, seq_len=None):
        if self.train_dist == "logit-normal":
            t = self.logit_normal_pdf(t)
        elif self.train_dist == "linear":
            pass
        else:
            raise ValueError()
    
        if self.train_shift:
            if self.static_mu:
                mu = self.static_mu
            elif seq_len:
                mu = self.lerp_mu(seq_len, self.min_seq_len, self.max_seq_len, self.min_mu, self.max_mu)
            else:
                raise ValueError()
            t = self.t_shift(mu, 1.0, t)
        return t

    def get_train_t(self, size, seq_len=None):
        t = torch.rand(size)
        t = self.lin_t_to_dist(t, seq_len=seq_len)
        return t

    def get_loss_weighting(self, t):
        if self.loss_weight_dist == "scaled_clipped_gaussian":
            w = self.scaled_clipped_gaussian(t)
        elif self.loss_weight_dist == "logit-normal":
            w = self.logit_normal_pdf(t)
        elif self.loss_weight_dist is None:
            w = torch.ones_like(t)
        else:
            raise ValueError()
        return w


    def get_inference_t(self, steps, strength=1.0, seq_len=None, clip_by_strength=True):
        if clip_by_strength:
            true_steps = max(1, int(strength * steps)) + 1
        else:
            true_steps = max(1, steps) + 1
        t = torch.linspace(strength, 0.0, true_steps)
        t = self.lin_t_to_dist(t, seq_len=seq_len)
        return t

    def inference_ode_step(self, noise_pred: torch.Tensor, latents: torch.Tensor, index: int, t_schedule: torch.Tensor):
        t = t_schedule[index]
        t_next = t_schedule[index + 1]
        d_t = t_next - t
        latents = latents + d_t * noise_pred
        return latents
