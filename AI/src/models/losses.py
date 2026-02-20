import torch
import torch.nn as nn

class SmoothCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        n = logits.size(1)
        logp = torch.log_softmax(logits, 1)
        y = torch.zeros_like(logp).fill_(self.eps / n)
        y.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return -(y * logp).sum(1).mean()