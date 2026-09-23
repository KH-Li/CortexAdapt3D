"""Focal loss used for dHCP binary classification."""

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        weight = torch.where(targets == 1, 1 - self.alpha, self.alpha)
        return (-weight * (1 - pt).pow(self.gamma) * log_pt).mean()
