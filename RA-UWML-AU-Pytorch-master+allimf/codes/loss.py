import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

EPS = 1e-7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UncertainLoss(nn.Module):

    def __init__(self, target_task, init_sigma):
        super(UncertainLoss, self).__init__()

        self.target_task = target_task
        if 'classify' in self.target_task:
            self.criterion_classify = nn.CrossEntropyLoss(reduction='none').to(device)
        if 'regress' in self.target_task:
            self.criterion_regress = nn.SmoothL1Loss(reduction='none').to(device)

        self.init_sigma = torch.tensor(init_sigma).to(device)
        self.softplus = torch.nn.Softplus()

    def forward(self, pred_mean, pred_std, pred_logits, targets, success):
        N = targets.size(0)

        # 【还原点】线性权重 (0.61版本用的这个)
        weights = 1.0 + (targets.float() ** 2)

        if 'classify' in self.target_task:
            pred_logits = pred_logits.view(N, 12, 6).view(-1, 6)
            targets_long = targets.long().view(-1)
            loss_logits = self.criterion_classify(pred_logits, targets_long)
            loss_logits = loss_logits.view(N, 12)

        if 'regress' in self.target_task:
            loss_mean = self.criterion_regress(pred_mean, targets.float())
            loss_mean = loss_mean * weights

        if 'uncertain' in self.target_task:
            pred_std = self.softplus(self.init_sigma + pred_std)

            # 【还原点】min=0.05
            # 注意：这是你0.61版本时已经加上了的，否则Loss会是负数
            pred_std = torch.clamp(pred_std, min=0.05, max=5.0)

            loss = loss_mean * 2 ** 0.5 / (pred_std + EPS) + (pred_std + EPS).log()
        else:
            loss = loss_mean

        if 'classify' in self.target_task:
            if 'regress' in self.target_task:
                loss = loss_logits + loss
            else:
                loss = loss_logits

        batch_loss = success.view(N, -1).float() * loss

        return batch_loss.mean()