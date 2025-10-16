import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils

import io
import wandb
from PIL import Image
import geometry

class DDPM:

    def __init__(self, betas):
        self.betas = betas
        self.a, self.abar = alpha_bar(betas)  # \bar{α}_t for t=1..T
        self.sqrt_a = torch.sqrt(self.a)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.abar)
        self.sqrt_abar = torch.sqrt(self.abar)

    def plot_alpha_beta(self):
        T = len(self.betas)
        plt.plot(range(1, T + 1), self.a, label=r"$'\alpha'_t$ (Linear Beta)")
        plt.plot(range(1, T + 1), self.betas, label=r"$'\beta'_t$ (Linear Beta)")
        plt.plot(range(1, T + 1), self.abar, label=r"$\bar{\alpha}_t$ (Linear Beta)")
        plt.xlabel("Timestep")
        plt.ylabel(r"$\bar{\alpha}_t$")
        plt.title("Alpha Bar vs Timestep (Linear Beta Schedule)")
        plt.grid(True)
        plt.legend()
        plt.show()



    def denoising_step(self, t, z_t, z_eps):
        alpha_t = self.a[t].unsqueeze(-1)
        beta_t = self.betas[t].unsqueeze(-1)
        abar_t = self.abar[t].unsqueeze(-1)
        abar_tm1 = self.abar[t - 1].unsqueeze(-1)
        abar_tm1[t==0] = abar_t[t==0]
        # print("A", z_t.shape, beta_t.shape, abar_t.shape, z_eps.shape)

        z_mean_tm1 = 1 / alpha_t.sqrt() * (z_t - (1 - alpha_t) / torch.clamp(1-abar_t, min=1e-20).sqrt() * z_eps)

        noise = torch.randn_like(z_t)
        noise[t == 0] = 0

        # Fixed variance choice
        z_var_tm1 = beta_t * (1 - abar_tm1) / torch.clamp(1.0 - abar_t, min=1e-20) # posterior variance (small)
        if False:  # high variance alternative (may lead to more diversity but lower quality)
            z_var_tm1 = beta_t.clamp(min=1e-20)

        z_tm1 = z_mean_tm1 + z_var_tm1.sqrt()*noise

        z_0_hat = (z_t - torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-20)) * z_eps) / torch.sqrt(abar_t)

        # if False: # rescale if norm>1 instead of clamp h_0
        #     norm = z_0_hat.norm(dim=-1, keepdim=True).clamp(min=1)
        #     z_0_hat = z_0_hat/norm
        #
        #     norm = z_tm1.norm(dim=-1, keepdim=True).clamp(min=1)
        #     z_tm1 = z_tm1/norm

        boundfn = lambda x : torch.clamp(x, -1, 1)

        if True:
            boundfn = lambda x : geometry.project_from_inside_cube(z_t, x)
            #boundfn = lambda x : geometry.project_from_inside_sphere(z_t, x)

        if True: # clamp
            return boundfn(z_tm1), z_mean_tm1, z_var_tm1, boundfn(z_0_hat),
        else: # does not train at all
            return z_tm1, z_mean_tm1, z_var_tm1, z_0_hat,

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        # forward diffusion process
        """
        x_t = sqrt(ā_t) x0 + sqrt(1 - ā_t) ε,  ε ~ N(0, I)
        x0: (B, C, H, W) or (B, D, ...)
        t : (B, 1) ints in [0, T]
        """
        B = x0.shape[0]
        eps = torch.randn_like(x0)
        abar_t = self.abar[t]
        x_t = abar_t.sqrt().unsqueeze(-1) * x0 + (1.0 - abar_t).sqrt().unsqueeze(-1) * eps # checked
        return x_t, eps


    # ---- Apply the trained model: full reverse sampling loop ----
    @torch.no_grad()
    def ddpm_sample(self, eps_model, z_1, conditions, noT=False):
        """
        Generate x0 samples with a fixed-variance DDPM.
        - eps_model: trained network, takes (x_t, t) and predicts ε
        - shape: (B,C,H,W) or (B,D,...) of desired samples
        - betas: (T,) schedule used at training
        - var_type: 'fixed_small' (posterior variance) or 'fixed_large' (β_t)
        - x_T: optional starting noise; if None, standard normal is used
        """
        eps_model.eval()

        T = self.betas.numel() - 1

        B = z_1.shape[0]
        z_t = z_1 # init
        intermediate_start = []
        intermediate_mean = []
        intermediate_end = []
        intermediate_x0 = []
        for t_int in range(T, 0, -1):  # T, T-1, ..., 1
            intermediate_start.append(z_t)
            # Scalars for this step (broadcasted automatically)
            beta_t = self.betas[t_int - 1]
            alpha = 1.0 - beta_t
            abar_t = self.abar[t_int]
            abar_tm1 = self.abar[t_int - 1]

            # Model prediction εθ(x_t, t)
            t_batch = torch.full((B,), t_int, dtype=torch.long)
            cond_list = conditions + [t_batch]
            if noT:
                cond_list = conditions
            eps_pred = eps_model(z_t, cond_list)["y"]

            # note, z_t updated/replaced, moving on to the next time step
            z_t, z_mean, z_var, z_0  = self.denoising_step(t_batch, z_t, eps_pred)

            if not utils.is_valid(eps_pred):
                print("Warning, encountered issue", eps_pred)


            if not utils.is_valid(z_mean):
                print("Warning, encountered issue", z_mean)

            intermediate_mean.append(z_mean)
            intermediate_end.append(z_t)
            intermediate_x0.append(z_0)

            if not utils.is_valid(z_t) or not utils.is_valid(z_0):
                print("NaN", t_int, beta_t, alpha, abar_t, abar_tm1, eps_pred, z_mean, z_t, z_0)
                return z_t

        return z_0, [intermediate_start, intermediate_mean, intermediate_end, intermediate_x0]  # this is x_0 (same scaling as your training data)


    def ddim_step(self, x_t, t, tau, eps_pred):
        """
        x_t: current sample at step t
        t: current index
        tau: target index (< t)
        eps_pred: model prediction eps_theta(x_t, t)
        abar: precomputed cumulative alphas (len T+1, abar[0]=1)
        """
        abar_t = self.abar[t]
        abar_tau = self.abar[tau]

        # Predict x0
        x0_pred = (x_t - torch.sqrt(1 - abar_t) * eps_pred) / torch.sqrt(abar_t)

        # Deterministic update
        x_tau = torch.sqrt(abar_tau) * x0_pred + torch.sqrt(1 - abar_tau) * eps_pred
        return x_tau

def alpha_bar(betas: torch.Tensor) -> torch.Tensor:
    """Compute \bar{α}_t = ∏_{i=1}^t (1 - β_i)."""
    alphas = 1.0 - betas
    alphasbar = torch.cumprod(alphas, dim=0)
    return alphas, alphasbar

# ---- Linear schedule (Ho et al. 2020) ----
def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2,
                         device=None, dtype=torch.float32) -> torch.Tensor:
    """β_t linearly spaced in [beta_start, beta_end]."""
    return torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)
