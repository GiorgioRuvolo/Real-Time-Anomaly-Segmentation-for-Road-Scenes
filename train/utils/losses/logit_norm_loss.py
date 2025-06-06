# =====================================================================================
# Mitigating Neural Network Overconfidence with Logit Normalization
# GitHub: https://github.com/hongxin001/logitnorm_ood/blob/main/common/loss_function.py
# =====================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitNormLoss(nn.Module):

    def __init__(self, loss=None, t=1.0):   # CrossEntropyLoss or FocalLoss
        super(LogitNormLoss, self).__init__()
        self.loss = loss    
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        # return F.cross_entropy(logit_norm, target)
        return self.loss(logit_norm, target)
    
    def __str__(self):
        return f"LogitNormLoss(loss={self.loss}, t={self.t})"