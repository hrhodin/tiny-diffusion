import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Iterator, Dict

import torch.optim as optim
import models, geometry, datasets, losses, ddpm
from torch.utils.data import Dataset, DataLoader

class Forward1DSq():
    def __init__(self, offset = 5):
        self.offset = offset # offset to make the latent variables have mean 0 and std 1


    def init(self, bs):
        return torch.randn(bs, 1)

    def extract(self, z):
        return self.offset+z

    def forward(self, z):
        x = self.extract(z)
        return x*x

    def plot_eps(self, samples, eps, title="Eps", loss=None):

        y1 = self.extract(samples)
        targets = samples + eps
        y2 = self.extract(targets)
        x = torch.arange(0, y1.shape[0], y1.shape[0])
        #ds, dd = s2-s1, d2-d1
        scale = 1
        if loss is not None:
            scale = loss
        plt.scatter(x, y1, s=scale*10, alpha=0.3)

        for i in range(len(x)): #
            plt.gca().annotate("", xytext=(x[i].item(), y1[i].item()), xy=(x[i].item(), y2[i].item()),
                        arrowprops=dict(arrowstyle="->"))
        plt.title(title)
        plt.show()

    def plot_samples(self, samples, title="Samples", buffer=None):
        plt.figure()
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)  # horizontal axis
        plt.axvline(0, color="black", linewidth=1, alpha=0.5)  # vertical axis

        y = self.extract(samples)
        x = torch.arange(0, y.shape[0], y.shape[0])

        plt.scatter(x, y, s=10, alpha=0.3)
        plt.xlabel("id")
        plt.ylabel("y")
        plt.title(title)
        plt.grid(True)

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
        self.project = Forward1DSq()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator(device="cpu").manual_seed(self.seed + idx)
        mean, std = 0, 1.0
        z0 = (mean+std*torch.randn(1, generator=g, dtype=self.dtype, device=self.device))
        p0 = self.project.forward(z0.unsqueeze(0)).squeeze(0)
        return {"p_0" : p0, "z_0": z0}
