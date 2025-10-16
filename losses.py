import math

import matplotlib.pyplot as plt
import numpy as np
import torch

import example_2Dsimple
import geometry
def circle_loss(p_t, p_0):
    iou = 1.0-geometry.circle_iou_concentric(p_t, p_0)
    return iou

# Plot
def plot_losses(losses, title=""):
    t = torch.arange(len(losses))
    plt.plot(t,losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss '+title)
    plt.grid(True)
    plt.gca().set_yscale('log')
    plt.show()