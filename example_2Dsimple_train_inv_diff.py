import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import models, geometry, datasets, losses, ddpm, example_2Dsimple
from torch.utils.data import Dataset, DataLoader
import importlib
from matplotlib import cm
import utils

import io
import wandb
from PIL import Image

import example_2Dsimple as example
import training

def main():
    torch.autograd.set_detect_anomaly(True, check_nan=True) # TODO: disable when not debugging

    wandb.login()
    project = "inv-diff-2Dsphere"
    config = {
        'epochs': 5, #10000,
        'lr': 0.01,
        'bs': 1024,
        'T': 1000,
    }
    config['dataset size'] = 1000*10000

    betas = ddpm.linear_beta_schedule(config['T'])
    diffusion = ddpm.DDPM(betas)
    diffusion.plot_alpha_beta()
    rand_conditions = 1 # set to 0 to disable
    model_diff = models.SimpleNN(IO=2, N=10, C=rand_conditions+1+1, normalize=False) # HACK

    # Forward model & optimizer
    optimizer = optim.Adam(model_diff.parameters(), lr=0.001)
    dataset = example.Dataset2DSphere(length=config['dataset size'], seed=8)
    forward = example.Forward2DSphere()
    train_loader = DataLoader(
        dataset,
        batch_size=config['bs'],
        shuffle=True,  # reshuffle each epoch
        num_workers=0,  # >0 for parallel loading
        pin_memory=False  # True if training on GPU
    )

    testset = example.Dataset2DSphere(length=config['bs'], seed=999)
    testloader = DataLoader(
        testset,
        batch_size=config['bs'],
        shuffle=False,  # reshuffle each epoch
        num_workers=0,  # >0 for parallel loading
        pin_memory=False  # True if training on GPU
    )

    # start training
    log = []
    with wandb.init(project=project, config=config) as run:
        for epoch in range(config['epochs']):
            if epoch != 0: # skip to validate once before training
                run.watch(model_diff, log_freq=100, log="all") # log gradients
                log_epoch = training.train_inv_epoch(
                                       model_diff, forward, diffusion, losses.circle_loss,
                                       train_loader, optimizer, run,
                                        RANDOMIZE_T_IN_SEG=False,
                                        disable_inv=True) # HACK
                log.extend(log_epoch)

                if len(log)>500:
                    losses.plot_losses(log[100:], f"epoch{epoch}")
                else:
                    losses.plot_losses(log, f"epoch{epoch}")
                run.unwatch()

            training.validate(testloader, forward, model_diff, diffusion, config)

    pass

if __name__ == '__main__':
    main()