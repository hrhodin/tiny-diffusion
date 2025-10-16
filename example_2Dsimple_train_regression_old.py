import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import models, geometry, datasets, losses, ddpm
from torch.utils.data import Dataset, DataLoader
import importlib
import example_2Dsimple
import utils

def main():
    model_reg = models.SimpleNN(IO=2, C=1, N=10, bound=False)
    bs = 1000

    # Loss & optimizer
    optimizer = optim.Adam(model_reg.parameters(), lr=0.01)

    dataset = datasets.Dataset2DSphere(10000*bs, seed=8)
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,  # reshuffle each epoch
        num_workers=0,  # >0 for parallel loading
        pin_memory=False  # True if training on GPU
    )

    # Minimalistic loop
    log = []
    for iter, batch in enumerate(loader):
        s_0, z_0, p_0, s_1, z_1, p_1 = batch['s_0'], batch['z_0'], batch['p_0'], batch['s_1'], batch['z_1'], batch['p_1']
#        s_0, z_0, p_0, x_t = batch['s_0'], batch['z_0'], batch['p_0'], batch['x_t']
        x_t = torch.cat([s_1, z_1], dim=-1)
        optimizer.zero_grad()  # reset gradients
        x_eps = model_reg(x_t, p_0)  # regression
        x_t = x_t+x_eps
        diff = losses.circle_loss(x_t[:, 0], x_t[:, 1], z_0, s_0)
        loss = diff.mean()
        loss.backward()  # backprop
        optimizer.step()  # update weights
        log.append(loss.item())
        if utils.print_at_logarithmic_steps(iter):
            print(f"Iter {iter + 1}, loss = {loss.item():.4f}")

    print(f"z={x_t[0]}, s={x_t[1]} proj{example_2Dsimple.project_1D(x_t[0], x_t[1])}, {-geometry.circle_iou_concentric(example_2Dsimple.project_1D(x_t[0], x_t[1]), s_0)}")

    losses.plot_losses(log)

    # plot starting point
    plt.plot(p_1); plt.show()
    samples_start = torch.concatenate([s_1,z_1], dim=-1)
    example_2Dsimple.plot_2d_samples(samples_start, p_1) # HACK

    # what if we feed that into a diffusion inference?
    T=100
    betas = ddpm.linear_beta_schedule(T)
    a, abar = ddpm.alpha_bar(betas)  # \bar{α}_t for t=1..T
    importlib.reload(utils)
    importlib.reload(ddpm)

    s_0 = torch.FloatTensor([1.0]) #torch.rand(1)  # 100 samples, 2 features
    z_0 = torch.FloatTensor([1.0]) # torch.rand(1)  # class labels {0,1}
    p_0 = example_2Dsimple.project_1D(z_0, s_0).expand(p_1.shape)
    x_1 = torch.randn(x_t.shape)
    samples, intermediates = ddpm.ddpm_sample(model_reg, x_1, [p_0], betas=betas, abar=abar, noT=True)
    print(samples)

    importlib.reload(example_2Dsimple)
    example_2Dsimple.plot_2d_samples(samples, p_0)
    #example_2Dsimple.plot_2d_samples(samples.abs(), p_0)

if __name__ == '__main__':
    main()