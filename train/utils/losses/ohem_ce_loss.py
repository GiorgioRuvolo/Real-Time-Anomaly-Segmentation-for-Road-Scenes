#!/usr/bin/python
# -*- encoding: utf-8 -*-

# Author: Coin Cheung
# Code: https://github.com/CoinCheung/BiSeNet/blob/master/lib/ohem_ce_loss.py
# Paper: https://arxiv.org/abs/1808.00897 (BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation)


import torch
import torch.nn as nn
import torch.nn.functional as F


#  import ohem_cpp
#  class OhemCELoss(nn.Module):
#
#      def __init__(self, thresh, lb_ignore=255):
#          super(OhemCELoss, self).__init__()
#          self.score_thresh = thresh
#          self.lb_ignore = lb_ignore
#          self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='mean')
#
#      def forward(self, logits, labels):
#          n_min = labels[labels != self.lb_ignore].numel() // 16
#          labels = ohem_cpp.score_ohem_label(
#                  logits, labels, self.lb_ignore, self.score_thresh, n_min).detach()
#          loss = self.criteria(logits, labels)
#          return loss


class OhemCELoss(nn.Module):
    """
    OhemCELoss (Online Hard Example Mining Cross Entropy Loss)

    This class implements the Ohem technique, a variant of the Multiclass Cross Entropy Loss.
    Instead of focusing on all pixels in a given batch, it calculates the loss considering 
    only the difficult pixels (e.g. pixels where the model performs poorly).

    In this implementation, hard examples are selected based on a given threshold. Only the 
    examples with a loss value greater than the threshold are used to compute the final loss.
    """

    def __init__(self, thresh=0.7, lb_ignore=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.lb_ignore = lb_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.lb_ignore].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)
    
    def __str__(self):
        return f"OhemCELoss(thresh={self.thresh.item()})"


if __name__ == '__main__':
    pass