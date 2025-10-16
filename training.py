import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import losses
import io

import example_2Dsimple
import geometry
from matplotlib import cm
import wandb
from PIL import Image


def train_inv_epoch(model, forward, diffusion, criterion, dataloader, optimizer, run,
                    epoch=0, RANDOMIZE_T_IN_SEG=False, disable_inv=False):
    # Minimalistic loop
    log = []
    for i, batch in enumerate(dataloader):
        p_0, z_0 = batch['p_0'], batch['z_0']
        bs = len(p_0)
        T = len(diffusion.betas)

        if disable_inv:
            z_0_hat = z_0
        else:
            # warmup
            with torch.inference_mode():
                model.eval()
                range = (250, 400)
                t_1 = torch.randint(range[0], range[1], (bs,))
                z_1 = forward.init(bs)
                rnd = torch.randn(bs)
                z_eps = model(z_1, [rnd, p_0, t_1])["y"]  # diffusion
                z_tm1, z_mean, z_var, z_0_hat = diffusion.denoising_step(t_1, z_1, z_eps)
        model.train()
        # TODO: may be necessary to run more sampling steps, like at actual inference

        # sample step times t and add noise up to step t
        if RANDOMIZE_T_IN_SEG:
            # select a random segment and then spread samples within
            N_seg = 10
            T_seg = T // N_seg
            T_off = torch.randint(0, N_seg, (1,)) * T_seg
            t = torch.randint(0, T_seg, (bs,)) + T_off
        else:
            t = torch.randint(0, T, (bs,))

        z_t, z_eps_gt = diffusion.q_sample(z_0_hat, t)
        rnd = torch.randn(bs)
        ret = model(z_t, [rnd, p_0, t])  # diffusion
        z_eps, aux_loss = ret["y"], ret["loss"]
        z_tm1, z_mean_hat, z_var_hat, z_0_hat = diffusion.denoising_step(t, z_t, z_eps)

        # deterministic process
        if disable_inv:
            #diff = (z_0_hat - z_0).pow(2).mean(dim=-1)
            diff = (z_eps - z_eps_gt).pow(2).mean(dim=-1)
        else:
            p_0_hat = forward.forward(z_0_hat)
            diff = criterion(p_0, p_0_hat)


        loss = diff.mean()
        optimizer.zero_grad()  # reset gradients

        loss.backward(retain_graph=True)  # backprop

        # Auxillary loss for mimicking bigger batch sizes etc.
        if aux_loss is not None:
            model.bn.track_gradients = False
            aux_loss.backward()
            model.bn.track_gradients = True

        optimizer.step()  # update weights

        run.log({"loss": loss})
        run.log({"aux_loss": aux_loss})
        if hasattr(model.bn, "bns"):
            for ci, bn in enumerate(model.bn.bns):
                run.log({f"running_mean_{ci}": bn.running_mean.mean()})
                run.log({f"running_var_{ci}": bn.running_var.mean()})
        if hasattr(model.bn, "running_mean"):
            run.log({"running_mean": model.bn.running_mean.mean()})
            run.log({"running_var": model.bn.running_var.mean()})
        if hasattr(model.bn, "running_mean_grad"):
            run.log({"running_mean_grad": model.bn.running_mean_grad.mean()})
            run.log({"running_var_grad": model.bn.running_var_grad.mean()})

        log.append(loss.item())
        if utils.print_at_logarithmic_steps(i) or i == len(dataloader) - 1:
            print(f"Iter {i + 1}, loss = {loss.item():.4f}")
            if i >= 99:  # draw samples
                with torch.inference_mode():
                    model.eval()

                    Nplot = 30
                    if False:
                        forward.plot_step(z_t[:Nplot], z_t[:Nplot] + z_eps_local[:Nplot], p_0[:Nplot],
                                          loss=diff[:Nplot], title=f"Eps at training iter{i}, epoch{epoch}")
                        plt.show()
                    else:  # arrows of this plot point in the same direction as the previous, but only a tiny step length
                        forward.plot_step(z_t[:Nplot], z_0_hat[:Nplot], p_0[:Nplot], loss=diff[:Nplot],
                                          title=f"z_mean at training iter{i}, epoch{epoch}")
                        plt.show()


                    rnd = torch.randn(bs)
                    z_1 = forward.init(bs)
                    samples, intermediates = diffusion.ddpm_sample(model, z_1, [rnd, p_0])
                    buffer = forward.plot_samples(samples, p=p_0,
                                                  title=f"DDPM Samples (aligned, iter{i}, epoch{epoch})",
                                                  buffer=io.BytesIO())
                    plt.show()
                    wandb.log({'inference samples': wandb.Image(Image.open(buffer), caption=f"Samples at i={i}")})
                    buffer2 = utils.plot_pred_hist(z_eps,
                                                   title=f"Prediction Histogram (training, iter{i}, epoch{epoch})",
                                                   buffer=io.BytesIO())
                    wandb.log({'pred histogram': wandb.Image(Image.open(buffer2), caption=f"Hist at i={i}")})
                    plt.show()
    return log

def validate(testloader, forward, model_diff, diffusion, config):
    with torch.inference_mode():
        model_diff.eval()

        # now sample from the learned (conditional) distribution
        # create condition (input)
        batch = next(iter(testloader))
        p_0, z_0 = batch['p_0'], batch['z_0']
        if False:  # static HACK
            z_0[:, 0] = 1
            z_0[:, 1] = 5
            p_0 = forward.forward(z_0)

        # plot starting point
        z_1 = forward.init(config['bs'])
        p_1 = forward.forward(z_1)
        if False:  # starting point
            plt.plot(p_1);
            plt.title("Starting point p_1");
            plt.show()
            forward.plot_samples(z_1, p=p_1 * 0 + 1, title="Start Samples (orig)")
            forward.plot_samples(z_1, p=p_1, title="Start Samples (aligned with self)")
            forward.plot_samples(z_1, p=p_0, title="Start Samples (aligned with p_0)")

        if True:  # distribution
            bins = None
            count_ts = []
            for t in torch.arange(0, config['T'], config['T'] / 100, dtype=int):
                t_i = torch.ones((config['bs'],), dtype=int) * t
                rnd = torch.randn(config['bs'])
                z_eps = model_diff(z_1, [rnd, p_0, t_i])["y"]
                color = cm.inferno(t / config['T'])
                counts, bins = utils.plot_pred_hist(z_eps, title=f"Eps prediction Histogram (eval) for t={t}", append=True,
                                                    color=color)
                count_ts.append(counts)

            from scipy.stats import moment
            moment_str = ""
            torch.set_printoptions(precision=2)
            moments = ["count", "mean", "var", "skew", "kurt"]
            for mi in range(5):
                m = moment(z_eps, moment=mi)
                moment_str += f" {moments[mi]}={m}"
            plt.gca().set_xlabel(moment_str + " (gt Gauss: 0,1,0,1,3)")
            plt.show()
            count_ts = torch.stack(count_ts)

            # summary over all
            counts_mean = torch.mean(count_ts, dim=0)
            counts_std = torch.std(count_ts, dim=0)
            x = (bins + (bins[1] - bins[0]) / 2)[:-1]  # compute midpoint of bin
            s = 2
            plt.plot(x, counts_mean, color="blue")
            plt.fill_between(x, counts_mean - s * counts_std, counts_mean + s * counts_std, color="blue", alpha=.1)
            plt.title("Mean and std of eps predictions (eval)")
            plt.show()

        if False:
            rnd = torch.randn(config['bs'])
            samples, intermediates = diffusion.ddpm_sample(model_diff, z_1, [rnd, p_0])
            forward.plot_samples(samples, p=p_1 * 0 + 1, title="DDPM Samples (orig, validation)")
            forward.plot_samples(samples, p=p_0, title="DDPM Samples (aligned, validation)")
        #    example_2Dsimple.plot_2d_samples(samples.abs(), p_0)

        # sample again to see the effect of randomness
        samples, intermediates = diffusion.ddpm_sample(model_diff, z_1, [rnd, p_0])

        forward.plot_samples(samples, p=p_1 * 0 + 1, title="DDPM Samples (orig, validation)")
        forward.plot_samples(samples, p=p_0, title="DDPM Samples (aligned, validation)")

        forward.plot_sample_trajectories(intermediates, p=p_0, N=1, T=200, title="DDPM trajectories (aligned, validation)")

        # take regular samples
        sigma = 1
        num_samples_on_axis = 10
        xs = torch.linspace(-sigma, sigma, steps=num_samples_on_axis)
        ys = torch.linspace(sigma, -sigma, steps=num_samples_on_axis)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        xy = torch.stack([x, y], dim=-1)
        z_t_regular = xy.view(-1, 2)

        # Test different time steps
        num_ax = 10
        num_x = math.ceil(math.sqrt(num_ax))
        num_y = math.floor(math.sqrt(num_ax))
        fig, axs_2d = plt.subplots(num_x, num_y, figsize=(10, 10))
        axs = [x for row in axs_2d for x in row]
        fig0, axs0_2d = plt.subplots(num_x, num_y, figsize=(10, 10))
        axs0 = [x for row in axs0_2d for x in row]
        for i in range(num_ax):
            # ax = subfig.subplots(1, 1)
            t_i_single = i * 1000 // num_ax
            t_i = torch.ones(len(z_t_regular), dtype=int) * t_i_single
            p_0 = torch.ones(len(z_t_regular))

            # plot loss landscape alongside
            p_0_hat = forward.forward(z_t_regular)
            diff = losses.circle_loss(p_0, p_0_hat)

            rnd = torch.randn(len(z_t_regular))
            eps_pred = model_diff(z_t_regular, [rnd, p_0, t_i])["y"]
            z_tm1, z_mean, z_var, z_0_hat = diffusion.denoising_step(t_i, z_t_regular, eps_pred)
            z_mean_amplified = z_t_regular + (z_mean - z_t_regular) * 10

            ax = axs[i]
            ax.set_aspect('equal', adjustable='box')
            forward.plot_step(z_t_regular, z_mean_amplified, p_0, ax=ax, loss=diff, title=f"Eps t={t_i_single}")

            ax = axs0[i]
            ax.set_aspect('equal', adjustable='box')
            forward.plot_step(z_t_regular, z_0_hat, p_0, ax=ax, loss=diff, title=f"z_0 t={t_i_single}")
        plt.show()