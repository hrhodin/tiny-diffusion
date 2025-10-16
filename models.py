import math

import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim


class ConditionalBatchNorm1D(nn.Module):
    def __init__(self, num_conditions, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()

        self.bns = [nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True) for i in range(num_conditions)]
    def forward(self, x, labels):
        groups = {}
        for lbl in labels.unique():
            idx = (labels == lbl).nonzero(as_tuple=True)[0]  # indices for this label
            groups[lbl.item()] = (idx, x[idx])  # (indices, images)

        out = torch.empty_like(x)
        for lbl in labels.unique():
            idx = (labels == lbl).nonzero(as_tuple=True)[0]
            out[idx] = self.bns[lbl](x[idx])
        return {"y": out, "loss" : None}

class RollingBatchNorm1D_Baseline(nn.BatchNorm1d):

    def __init__(self, num_features, virtual_batch_size, momentum=0.1, eps=1e-5, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        #self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        y = super().forward(x)
        return {'y': y, 'loss': None}

class RollingBatchNorm1D(nn.Module):

    def __init__(self, num_features, virtual_batch_size, H, eps=1e-5, affine=True, track_running_stats=True):
        super().__init__()

        #self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.virtual_batch_size = virtual_batch_size
        self.num_features = num_features
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.running_mean_grad = torch.zeros(num_features)
        self.running_var_grad = torch.ones(num_features)
        self.eps = eps
        self.H = H
        self.track_gradients = True

    def momentum_to_hlen(momentum, batch_size):
        H_plus_batch = batch_size / momentum
        H = H_plus_batch - batch_size
        return H
    def forward(self, x):
        if not self.training:
            return {"y": (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)}
            # TODO: add learnable paras

        bs = len(x)
        badd = self.virtual_batch_size - bs

        # a_pos = torch.ones([badd//2,self.num_features], device=x.device) * self.bn.running_mean + self.bn.running_var.sqrt()
        # a_neg = torch.ones([badd-badd//2,self.num_features], device=x.device) * self.bn.running_mean+ self.bn.running_var.sqrt()
        # x_extended = torch.cat((x, a_pos, a_neg), dim=0)

        H = self.H
        def mean_hook(g):
            if self.track_gradients:
                with torch.no_grad():
                    self.running_mean_grad = (H * self.running_mean_grad + g * bs) / (H + bs)

        def var_hook(g):
            if self.track_gradients:
                with torch.no_grad():
                    self.running_var_grad = (H * self.running_var_grad + g * bs) / (H + bs)

        # update running stats
        mean_real = x.mean(dim=0)
        var_real  = x.var(dim=0, unbiased=False)
        mean_real.register_hook(mean_hook)
        var_real.register_hook(var_hook)

        mean_virtual = (bs * mean_real + badd * self.running_mean) / (bs + badd)
        var_virtual  = (bs * var_real + badd * self.running_var) / (bs + badd)

        with torch.no_grad():
            self.running_mean = (H * self.running_mean + bs * mean_real) / (H+bs)
            self.running_var  = (H * self.running_var  + bs * var_real) / (H+bs)

        # Weight relative to the real and virtual batch size part badd
        virtual_loss = torch.sum(self.running_var_grad[None, :] * var_real + self.running_mean_grad[None, :] * mean_real)
        virtual_loss = virtual_loss * badd / bs
        # Note, badd could be negative, leading to an effectively smaller batch size for batch norm (TODO: test if any benefits)
        #print("virtual_loss", virtual_loss)
        xnorm = (x - mean_virtual) / torch.sqrt(var_virtual + 1e-5)

        # TODO: leaned offset
        return {"y": xnorm, "loss": virtual_loss}

def timestep_embedding(timesteps, dims):
    """
    Create sinusoidal timestep embeddings.
    timesteps: 1-D Tensor of timesteps (int or float)
    dim: embedding dimension
    """
    half_dim = dims // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    )
    args = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding.view(timesteps.shape[0],-1)

class SimpleNN(nn.Module):
    def __init__(self, IO, N, C=0, embedding_expand=32, bound=False, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.bound = bound
        self.embedding_expand = embedding_expand
        self.fc1 = nn.Linear(IO+C*embedding_expand, N)  # input layer: 2 → N
        self.fc2 = nn.Linear(N, IO)  # output layer: N → 2
        #self.bn = nn.BatchNorm1d(num_features=IO, affine=False, track_running_stats=True)
        hlen = RollingBatchNorm1D.momentum_to_hlen(0.1,  batch_size=1024) # 1024 overshoots sometimes, 256 even more, 5000?
        # HACK self.bn = RollingBatchNorm1D(num_features=IO, virtual_batch_size=512, H=hlen, affine=False, track_running_stats=True)
        #self.bn = RollingBatchNorm1D_Baseline(num_features=IO, virtual_batch_size=512, affine=False, track_running_stats=True)
        self.bn = ConditionalBatchNorm1D(num_conditions=10, num_features=IO, affine=False, track_running_stats=True)
    def forward(self, x, conditions=None):
        if conditions is not None:
            if type(conditions) is list:
                combined = torch.cat([tin.unsqueeze(-1) if len(tin.shape)==1
                           else tin for tin in conditions], dim = -1)
            else:
                combined = conditions
            t_emb = timestep_embedding(combined, dims=self.embedding_expand)
            x = torch.concat([x, t_emb], dim=-1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        #x = torch.relu(x)

        if self.normalize:
            ret = self.bn(x, conditions[-1]//100) # self.bn(x.unsqueeze(1).unsqueeze(-1)).view(x.shape)
            #x = (x-x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 0.00001)
        else:
            ret = {"y": x, "loss": None}

        if self.bound:
            ret["y"] = torch.tanh(ret["y"])
        return ret
