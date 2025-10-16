import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Iterator, Dict

import torch.optim as optim
import models, geometry, datasets, losses, ddpm
from torch.utils.data import Dataset, DataLoader

# s_0/z_0 = p_0
#
# s_0 = 2
# p_0 = 2
# s_1 = 4
# z_1 = 2
#
# => normalized: s_1 = 2
#
# s_t/z_t / p_0  = (s_t/s_0) / (z_t / z_0)

class Forward2DSphere():
    def __init__(self, offset = 2.0, eps=1e-12):
        self.small_eps = eps
        self.offset = offset # offset to make the latent variables have mean 0 and std 1


    def init(self, bs):
        return torch.rand(bs, 2)-0.5

    def extract(self, z):
        return self.offset+z[:, 0], self.offset+z[:, 1]

    def forward(self, z):
        s, d = self.extract(z)
        return s.abs() / d.abs().clamp_min(self.small_eps)

    def plot_step(self, start, end, p, title="Eps", ax=None, loss=None, alpha=1):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot([1, 3], [1, 3], 'r',) # gt line
        s1, d1 = self.extract(start)
        s2, d2 = self.extract(end)

        # normalize scale size # p = s/d,   s=p*d
        s1, s2 = s1/p, s2/p

        if loss is not None:
            scale = loss
            ax.scatter(s1, d1, s=scale*10, alpha=0.3)

        for i in range(len(p)): #
            ax.annotate("", xytext=(s1[i].item(), d1[i].item()), xy=(s2[i].item(), d2[i].item()),
                        arrowprops=dict(arrowstyle="->",alpha=alpha))
        ax.set_title(title)

    def plot_samples(self, samples, p, title="Samples", buffer=None):
        plt.figure()
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)  # horizontal axis
        plt.axvline(0, color="black", linewidth=1, alpha=0.5)  # vertical axis

        s, d = self.extract(samples)
        s_normalized = s/p.squeeze()
        d_normalized = d
        plt.scatter(s_normalized, d_normalized, s=10, alpha=0.3)
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        std_s = s_normalized.std()
        mean_s = s_normalized.mean()
        std_z = d_normalized.std()
        mean_z = d_normalized.mean()
        n_std = 2
        plt.xlim([-n_std*std_s+mean_s, +n_std*std_s+mean_s])
        plt.ylim([-n_std*std_z+mean_z, +n_std*std_z+mean_z])
        plt.title(title)
        plt.grid(True)

        if buffer is not None:
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            return buffer
        else:
            plt.show()


    def plot_sample_trajectories(self, intermediates, p, N=10, T=100, title="Samples", buffer=None):
        plt.figure()
        ax = plt.gca()
        start, means, end, z0s = intermediates
        start = torch.stack(start)
        means = torch.stack(means)
        end = torch.stack(end)
        for t in range(min(T,len(means))):
            self.plot_step(start[t][:N], means[t][:N], p[:N], title="", ax=ax)
            self.plot_step(means[t][:N], end[t][:N], p[:N], title="", ax=ax, alpha=1-t/T)
            self.plot_step(start[t][:N], z0s[t][:N], p[:N], title="", ax=ax, alpha=0.2)

        plt.xlim([0.9, 3.5])
        plt.ylim([0.9, 3.1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(title)

        if buffer is not None:
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            return buffer
        else:
            plt.show()

from torch.utils.data import Dataset, DataLoader
class Dataset2DSphere(Dataset):
    def __init__(self, length: int, seed: int, device="cpu", dtype=torch.float32):
        self.length = int(length)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self.project = Forward2DSphere()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator(device="cpu").manual_seed(self.seed + idx)
        if True: # uniform
            offset, spread = -0.5, 1
            s0 = (offset+spread*torch.rand(1, generator=g, dtype=self.dtype, device=self.device))
            d0 = (offset+spread*torch.rand(1, generator=g, dtype=self.dtype, device=self.device))
        else:
            mean, std = 0, 0.3333
            s0 = (mean+std*torch.rand(1, generator=g, dtype=self.dtype, device=self.device))
            d0 = (mean+std*torch.rand(1, generator=g, dtype=self.dtype, device=self.device))

        #d0 = s0 # HACK
        z0 = torch.concatenate([s0,d0], dim=-1)
        p0 = self.project.forward(z0.unsqueeze(0)).squeeze(0)
        return {"p_0" : p0, "z_0": z0}
