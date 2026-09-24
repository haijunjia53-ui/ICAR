from __future__ import print_function
import os
import sys
import time
import logging
import torch
import csv
import numpy as np

term_width = 10

TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _pre_process(y_hat, y_lab):
    y_hat = np.array(y_hat, dtype=np.float64).T
    y_lab = np.array(y_lab, dtype=np.float64).T

    assert np.all(y_hat.shape == y_lab.shape)
    if len(y_hat.shape) == 1:
        y_hat = np.expand_dims(y_hat, axis=0)
        y_lab = np.expand_dims(y_lab, axis=0)
    return y_hat, y_lab


def nMAE(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    return np.mean(np.abs(y_hat-y_lab), 1)


def nMSE(y_hat, y_lab):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    return np.mean((y_hat - y_lab)** 2, 1)


def nICC(y_hat, y_lab, cas=3, typ=1):
    y_hat, y_lab = _pre_process(y_hat, y_lab)
    Y = np.array((y_lab, y_hat))
    # number of targets
    n = Y.shape[2]
    # mean per target
    mpt = np.mean(Y, 0)
    # print mpt.eval()
    mpr = np.mean(Y, 2)
    # print mpr.eval()
    tm = np.mean(mpt, 1)
    # within target sum sqrs
    WSS = np.sum((Y[0] - mpt) ** 2 + (Y[1] - mpt) ** 2, 1)
    # within mean sqrs
    WMS = WSS / n
    # between rater sum sqrs
    RSS = np.sum((mpr - tm) ** 2, 0) * n
    # between rater mean sqrs
    RMS = RSS
    # between target sum sqrs
    TM = np.tile(tm, (y_hat.shape[1], 1)).T
    BSS = np.sum((mpt - TM) ** 2, 1) * 2
    # between targets mean squares
    BMS = BSS / (n - 1)
    # residual sum of squares
    ESS = WSS - RSS
    # residual mean sqrs
    EMS = ESS / (n - 1)

    # --- 修改开始：添加 numpy 错误状态管理 ---
    with np.errstate(divide='ignore', invalid='ignore'):
        if cas == 1:
            if typ == 1:
                res = (BMS - WMS) / (BMS + WMS)
            if typ == 2:
                res = (BMS - WMS) / BMS
        if cas == 2:
            if typ == 1:
                res = (BMS - EMS) / (BMS + EMS + 2 * (RMS - EMS) / n)
            if typ == 2:
                res = (BMS - EMS) / (BMS + (RMS - EMS) / n)
        if cas == 3:
            if typ == 1:
                res = (BMS - EMS) / (BMS + EMS)
            if typ == 2:
                res = (BMS - EMS) / BMS
    # --- 修改结束 ---

    # 将无效值（NaN）和无穷大（Inf）都置为 0
    res[np.isnan(res)] = 0
    res[np.isinf(res)] = 0

    return res.astype('float32')

def _to_numpy(x):
    # Tensor -> numpy
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # 已经是 numpy，直接返回
    if isinstance(x, np.ndarray):
        return x
    # list / tuple 等，强制转 numpy
    return np.array(x, dtype=np.float64)


# --- 在 utils.py 顶部添加 import ---
from sklearn.metrics import f1_score

def pcc_paper(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """论文 PCC: sum (xi-x̄)(yi-ȳ) / ((n-1)SxSy)"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 2 or y.size != n:
        return 0.0

    x0 = x - x.mean()
    y0 = y - y.mean()

    sx = np.sqrt((x0 * x0).sum() / max(1, (n - 1)))
    sy = np.sqrt((y0 * y0).sum() / max(1, (n - 1)))

    denom = (n - 1) * sx * sy + eps
    return float((x0 * y0).sum() / denom)


def evaluate_pcc(prediction, target):
    """
    返回每个 AU 的 PCC（shape=[12]），与 test_ck+ 论文 PCC 一致的公式
    """
    prediction = _to_numpy(prediction)
    target = _to_numpy(target)

    # 保持和 evaluate_au 一样：先 clip 到 0-5
    prediction = np.clip(prediction, 0, 5)

    # prediction/target: [N, 12]
    assert prediction.shape == target.shape, f"shape mismatch: {prediction.shape} vs {target.shape}"

    pcc_list = []
    for k in range(prediction.shape[1]):
        pcc_list.append(pcc_paper(prediction[:, k], target[:, k]))
    return np.asarray(pcc_list, dtype=np.float32)


def evaluate_au(prediction, target):
    # 统一转成 numpy
    prediction = _to_numpy(prediction)
    target = _to_numpy(target)

    # 1. 确保预测值在合理范围内 (0-5)
    prediction = np.clip(prediction, 0, 5)

    # 2. 计算基础指标 (ICC 和 全体 MAE)
    MAE = nMAE(prediction, target)
    # MSE = nMSE(prediction, target) # 暂时不用
    ICC = nICC(prediction, target)

    # 3. 计算二分类 F1-Score (检测指标)
    # 逻辑：强度 >= 1 视为激活(Active)，< 1 视为未激活(Neutral)
    # 这能反映模型是否成功检测到了 AU 的发生，而不被大量的 0 样本淹没
    pred_bin = (prediction >= 1.0).astype(int)
    target_bin = (target >= 1.0).astype(int)

    F1_list = []
    for i in range(prediction.shape[1]):
        # zero_division=0 防止分母为0报错
        f1 = f1_score(target_bin[:, i], pred_bin[:, i], zero_division=0)
        F1_list.append(f1)
    F1 = np.array(F1_list)

    # 4. 计算正样本 MAE (MAE+) (强度准确性指标)
    # 逻辑：只计算 Ground Truth >= 1 的样本的 MAE
    # 这能反映：当真的有表情时，模型预测的强度到底有多准
    MAE_plus_list = []
    for i in range(prediction.shape[1]):
        active_indices = np.where(target[:, i] >= 1)[0]
        if len(active_indices) > 0:
            diff = np.abs(prediction[active_indices, i] - target[active_indices, i])
            MAE_plus_list.append(np.mean(diff))
        else:
            # 该 AU 在当前 batch 或验证集中没有激活样本
            MAE_plus_list.append(0.0)
    MAE_plus = np.array(MAE_plus_list)

    return MAE, ICC, F1, MAE_plus


def save_checkpoint(save_dir, iter_num, net):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state = {
        'net_state_dict': net.state_dict()
    }
    save_file = os.path.join(save_dir, 'ckpt_epoch_{}.pth'.format(iter_num))
    torch.save(state, save_file)


def save_loss(save_dir, valid_type, print_value):
    save_path = os.path.join(save_dir, valid_type + '.csv')

    csvfile = open(save_path, 'a', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(print_value)
    csvfile.close()


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


if __name__ == '__main__':
    pass
