import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

EPS = 1e-7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class L2HingeRankLoss(nn.Module):
    def __init__(self, margin_base=0.5, margin_slope=0.2):
        super().__init__()
        self.m0 = margin_base  # 基础门槛
        self.m1 = margin_slope  # 动态门槛系数

    def forward(self, pred, target, success, groups):
        # pred, target: [N, AU_num]
        # groups: [N] (即 subject_id)
        # success: [N] 或 [N, 1]

        total_loss = torch.tensor(0.0, device=pred.device)
        N, K = target.shape
        groups = groups.view(-1)
        valid_cnt = 0

        # =========== 【修复开始】处理 success 维度问题 ============
        # 如果 success 是 1D Tensor [N]，升维变成 [N, 1]
        if success.dim() == 1:
            success = success.view(-1, 1)
        # =========== 【修复结束】 ===============================

        for k in range(K):
            # 1. 挑出检测成功的帧
            # 如果 success 只有 1 列 (代表整张脸检测成功)，则所有 AU 共用
            # 如果 success 有 K 列 (代表每个 AU 单独的可见性)，则取第 k 列
            if success.shape[1] == 1:
                mask = success[:, 0] > 0.5
            else:
                mask = success[:, k] > 0.5

            idx = torch.nonzero(mask).view(-1)
            if idx.numel() < 2: continue

            p_k = pred[idx, k]
            y_k = target[idx, k]
            g_k = groups[idx]

            # 2. 利用广播机制构建矩阵 (Matrix Broadcasting)
            # Diff[i,j] = Val[i] - Val[j]
            y_diff = y_k.unsqueeze(1) - y_k.unsqueeze(0)  # 标签差
            p_diff = p_k.unsqueeze(1) - p_k.unsqueeze(0)  # 预测差

            # 3. 筛选配对：
            # (A) 同一个人: g[i] == g[j]
            # (B) 标签真的比它强: y[i] > y[j] + 0.1
            g_match = g_k.unsqueeze(1) == g_k.unsqueeze(0)
            pair_mask = g_match & (y_diff > 0.1)

            if not pair_mask.any(): continue

            # 4. 计算 Loss
            # 想要: p_diff > margin
            # 违例: margin - p_diff > 0
            margin = self.m0 + self.m1 * y_diff[pair_mask]
            violation = torch.relu(margin - p_diff[pair_mask])

            # 平方惩罚，标签差异越大权重越大
            loss_val = (violation ** 2) * (y_diff[pair_mask])
            total_loss += loss_val.mean()
            valid_cnt += 1

        return total_loss / (valid_cnt + 1e-6)

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