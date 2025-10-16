import math

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_pred_hist(z_eps, title="DefaultTitle", append=False, color='blue', buffer=False):
    # Plot histogram of predictions (from training)
    counts, bins = np.histogram(z_eps.detach().view(-1), bins=30, range=(-4, 4), density=True)
    plt.stairs(counts, bins, fill=not append, color=color)
    #plt.hist(, , )
    # parameters
    mu, sigma = 0.0, 1.0
    x = torch.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y = (1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    plt.plot(x.numpy(), y.numpy(), linewidth=2)
    plt.title(title)
    if not append:
        if buffer is not None:
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plt.show()
            return buffer
        else:
            plt.show()
        plt.show()
    return torch.from_numpy(counts), torch.from_numpy(bins)

def print_at_logarithmic_steps(iter):
    reference = math.pow(10.0, math.floor(math.log10(iter + 1)))
    return (iter + 1) % reference == 0

def is_valid(x):
    return not torch.isnan(x).any() and torch.isfinite(x).all()

def alpha_bar(betas: torch.Tensor) -> torch.Tensor:
    """Compute \bar{α}_t = ∏_{i=1}^t (1 - β_i)."""
    alphas = 1.0 - betas
    alphasbar = torch.cumprod(alphas, dim=0)
    return alphas, alphasbar

# ---- Linear schedule (Ho et al. 2020) ----
def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=10e-2,
                         device=None, dtype=torch.float32) -> torch.Tensor:
    """β_t linearly spaced in [beta_start, beta_end]."""
    return torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)


def denoising_step(a, betas, abar, t, z_t, z_eps):
    alpha_t = a[t].unsqueeze(-1)
    beta_t = betas[t].unsqueeze(-1)
    abar_t = abar[t].unsqueeze(-1)
    abar_tm1 = abar[t - 1].unsqueeze(-1)

    # print("A", z_t.shape, beta_t.shape, abar_t.shape, z_eps.shape)
    z_mean = (z_t - (beta_t / torch.sqrt(1.0 - abar_t)) * z_eps) / torch.sqrt(alpha_t)
    z_var = beta_t * (1.0 - abar_tm1) / (1.0 - abar_t + 1e-20)

    return z_mean, z_var

def q_sample(abar, x0: torch.Tensor, t: torch.Tensor):
    """
    x_t = sqrt(ā_t) x0 + sqrt(1 - ā_t) ε,  ε ~ N(0, I)
    x0: (B, C, H, W) or (B, D, ...)
    t : (B, 1) ints in [0, T]
    """
    B = x0.shape[0]
    eps = torch.randn_like(x0)
    abar_t = abar[t]
    x_t = abar_t.sqrt().unsqueeze(-1) * x0 + (1.0 - abar_t).sqrt().unsqueeze(-1) * eps
    return x_t, eps

#
# # ---- Apply the trained model: full reverse sampling loop ----
# @torch.no_grad()
# def ddpm_sample(eps_model, x_1, conditions, betas, abar, noT=False):
#     """
#     Generate x0 samples with a fixed-variance DDPM.
#     - eps_model: trained network, takes (x_t, t) and predicts ε
#     - shape: (B,C,H,W) or (B,D,...) of desired samples
#     - betas: (T,) schedule used at training
#     - var_type: 'fixed_small' (posterior variance) or 'fixed_large' (β_t)
#     - x_T: optional starting noise; if None, standard normal is used
#     """
#     eps_model.eval()
#
#     T = betas.numel() - 1
#
#     B = x_1.shape[0]
#     x = x_1 # init
#     for t_int in range(T, 0, -1):  # T, T-1, ..., 1
#         # Scalars for this step (broadcasted automatically)
#         beta = betas[t_int - 1]
#         alpha = 1.0 - beta
#         abar_t = abar[t_int]
#         abar_tm1 = abar[t_int - 1]
#
#         # Model prediction εθ(x_t, t)
#         t_batch = torch.full((B,), t_int, dtype=torch.long)
#         cond_list = conditions + [t_batch]
#         if noT:
#             cond_list = conditions
#         eps_pred = eps_model(x, cond_list)
#
#         # Mean of pθ(x_{t-1} | x_t)
#         mean = (x - (beta / torch.sqrt(1.0 - abar_t)) * eps_pred) / torch.sqrt(alpha)
#
#         # Fixed variance choice
#         var = beta * (1.0 - abar_tm1) / (1.0 - abar_t)  # \tilde{β}_t
#
#         # Add noise except at the final step to x0
#         if t_int > 1:
#             x = mean + torch.sqrt(var) * torch.randn_like(x)
#         else:
#             x = mean
#
#         if torch.isnan(x).any() or not torch.isfinite(x).all():
#             print("NaN", t_int, beta, alpha, abar_t, abar_tm1, eps_pred, mean)
#             return x

    return x  # this is x_0 (same scaling as your training data)