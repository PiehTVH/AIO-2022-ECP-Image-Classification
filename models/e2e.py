import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from copy import deepcopy


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.fe = timm.create_model(
                                    config.name,
                                    pretrained=False,
                                    num_classes=0,  # remove classifier nn.Linear
                                    )
        self.mlp = nn.Sequential(
            nn.Linear(config.channel, config.channel),
            nn.LeakyReLU(),
            nn.Linear(config.channel, 1),
            )
    
    def forward(self, x):
        x = self.fe.forward_features(x)                 # N, C, H, W
        x = torch.flatten(x, 2)                         # N, C, S = H * W
        x = torch.div(x, torch.clamp(torch.norm(x, dim=1, keepdim=True), min=0.00000001))

        a = self.mlp(torch.transpose(x, 1, 2))           # N, S, 1
        a = F.softmax(a, dim=1)
        a = torch.squeeze(a, dim=2)

        return x, a
    
    def select_feature(self, x, thres=0.95):
        with torch.no_grad():
            f, a = self.forward(x)
            mask = deepcopy(a)
            mask, indices = torch.sort(mask, dim=-1, descending=True)
            mask = torch.cumsum(mask, dim=-1)
            mask = torch.where(mask <= thres, 1.0, 0.0)
            a.scatter_(dim=1, index=indices, src=mask, reduce='multiply')
            a = torch.where(a > 0, 1.0, 0.0)
        return f, a
