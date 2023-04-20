import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from copy import deepcopy


class DimensionReductor(nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()
        # self.config = config
        # self.ln = nn.utils.parametrizations.orthogonal(nn.Linear(config.channel, config.model.target_dim, bias=False))
        self.ln = nn.utils.parametrizations.orthogonal(nn.Linear(512, 64, bias=False))

    def forward(self, x):
        x = self.ln(torch.transpose(x, -2, -1))                 # N, S, D
        x = torch.div(x, torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-10))
        return x
