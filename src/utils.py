
import numpy as np
import torch

def mulaw_baseline(x, mu=255.0):
    """Baseline G.711 Mu-Law standard per confronto"""
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    x_q = np.round(((x_mu + 1) / 2) * mu)
    x_q = np.clip(x_q, 0, mu)
    x_unq = (x_q / mu) * 2 - 1
    return np.sign(x_unq) * (1 / mu) * ((1 + mu)**np.abs(x_unq) - 1)