import os

# 【修复1】解决 DLL 冲突，必须放在 import torch 之前
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import json
import model_R8_AffectAU as model
from params import ParamsControl
from dataset_disfa import get_data_loader
from loss import UncertainLoss, L2HingeRankLoss
import utils
import torchvision.utils as vutils


# 【修复2】补回丢失的 setup_seed 函数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'[INFO] Random Seed set to: {seed}')


# 【修复3】补回丢失的全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iter_num = 0

best_valid_score_ = 0.0
best_valid_mae_ = 100000.


def main(opts):
    setup_seed(42)

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    opts.save_dir = os.path.join(opts.save_dir, opts.snapshot)
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    logging = utils.init_log(os.path.join(opts.save_dir, '{}.log'.format(opts.snapshot)))
    _print = logging.info
    _print(opts.snapshot)

    opts.train_num = '{:0>4}'.format(opts.train_num)
    opts.save_dir = os.path.join(opts.save_dir, opts.train_num)
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    _print(opts.train_num)

    with open(os.path.join(opts.save_dir, 'opts_setting.txt'), 'w') as f:
        json.dump(opts.__dict__, f, indent=2)

    # Dataset 加载
    train_loader, n_data_train = get_data_loader(opts, 'train', opts.fold)
    valid_loader, n_data_valid = get_data_loader(opts, 'valid', opts.fold)
    _print('Train Dataset Info')
    _print('train total samples:  {}'.format(n_data_train))
    _print('train batch_size:     {}'.format(opts.batch_size_train))
    _print('train batch_num:      {}'.format(len(train_loader)))
    _print('total {}*{} = {} iters'.format(len(train_loader), opts.last_epoch, len(train_loader) * opts.last_epoch))
    _print('Valid Dataset Info')
    _print('valid total samples:  {}'.format(n_data_valid))
    _print('valid batch_size:     {}'.format(opts.batch_size_valid))
    _print('valid batch_num:      {}'.format(len(valid_loader)))
    _print('total {}*{} = {} iters'.format(len(valid_loader), opts.last_epoch, len(valid_loader) * opts.last_epoch))

    net = model.AU_IMF_Net(opts).to(device)

    params = ParamsControl(opts, net)
    criterion = UncertainLoss(opts.target_task, opts.init_sigma).to(device)
    rank_criterion = L2HingeRankLoss(margin_base=0.5, margin_slope=0.2).to(device)

    AU_ID = ("AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17", "AU20", "AU25", "AU26")
    utils.save_loss(opts.save_dir, 'train', ("iter_num", "losses", "iccs", "maes"))
    utils.save_loss(opts.save_dir, 'valid', ("iter_num", "icc", "mae") + AU_ID * 2)

    for epoch in range(opts.start_epoch, opts.last_epoch):
        params.update(epoch)
        # 将 epoch 传入 train 函数
        train(epoch, train_loader, valid_loader, net, criterion, rank_criterion, params, opts)

    _print('valid best_valid_score:{}'.format(best_valid_score_))
    _print('valid best_valid_mae:{}'.format(best_valid_mae_))
    _print('finish...')
    _print('------' * 20)


def train(epoch, train_loader, valid_loader, net, criterion, rank_criterion, params, opts):
    global iter_num

    losses = utils.AverageMeter()
    maes = utils.AverageMeter()
    iccs = utils.AverageMeter()
    align_losses = utils.AverageMeter()

    net.train()

    # ================= 【策略修改】Warm-up 策略 =================
    # 如果是第 0 个 Epoch，强制不使用 Rank Loss，先让网络学好回归
    if epoch < 1:
        current_lambda_rank = 0.0
    else:
        current_lambda_rank = opts.lambda_rank
    # ==========================================================

    for i, data in enumerate(train_loader):
        img, ref_img, flow, label, success, subject_id = \
            data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[
                5].to(device)

        N = img.shape[0]

        # 前向传播
        pred_mean, pred_std, pred_logits, est_flow, aligned_ref, f_c = net(img, ref_img, flow)

        # 1. 任务 Loss
        loss_main = criterion(pred_mean, pred_std, pred_logits, label, success)

        # 计算 Rank Loss (根据 Warm-up 策略决定是否计算)
        if current_lambda_rank > 0:
            loss_rank = rank_criterion(pred_mean, label, success, groups=subject_id)
        else:
            loss_rank = torch.tensor(0.0).to(device)

        # 2. 对齐 Loss
        loss_align = torch.tensor(0.0).to(device)

        if opts.align_mode == 'implicit':
            loss_align_mse = F.mse_loss(aligned_ref, f_c.detach())

            if est_flow is not None:
                dy = torch.abs(est_flow[:, :, 1:, :] - est_flow[:, :, :-1, :]).mean()
                dx = torch.abs(est_flow[:, :, :, 1:] - est_flow[:, :, :, :-1]).mean()
                loss_smooth = (dx + dy)
            else:
                loss_smooth = 0.0

            loss_align = 10.0 * loss_align_mse + 0.1 * loss_smooth

        # 3. 总 Loss (使用动态调整的 rank 权重)
        loss = loss_main + current_lambda_rank * loss_rank + loss_align

        MAE, ICC, F1, MAE_plus = utils.evaluate_au(pred_mean, label)

        params.zero_grad()
        loss.backward()
        params.back_grad()

        losses.update(loss.item(), N)
        maes.update(MAE.mean(), N)
        iccs.update(ICC.mean(), N)
        align_losses.update(loss_align.item(), N)

        iter_num += 1
        if iter_num % opts.print_freq == 0:
            lrs = [group['lr'] for group in params.optimizers[0].param_groups]
            lr_str = f"LR_backbone={lrs[0]:.2e}" if len(lrs) > 0 else ""

            # 打印信息中显示当前的 Rank 权重状态
            print('{0}_Iter: [{1}][{2}]\t'
                  'Loss {losses.val:.3f} (Reg:{reg:.3f}, Rank({w_rank}):{rank:.3f}, Align:{align:.4f})\t'
                  'ICC {iccs.val:.3f} ({iccs.avg:.3f})\t'
                  'MAE {maes.val:.3f} ({maes.avg:.3f})\t'
                  '{lr}'.
                  format('Train', iter_num, epoch + 1,
                         losses=losses,
                         reg=loss_main.item(),
                         w_rank=current_lambda_rank,
                         rank=loss_rank.item(),
                         align=align_losses.val,
                         iccs=iccs, maes=maes, lr=lr_str))

            utils.save_loss(opts.save_dir, 'train', (iter_num, losses.avg, iccs.avg, maes.avg))

        if iter_num % opts.valid_freq == 0:
            valid('valid', epoch, valid_loader, net, opts)


def perform_smoothing(preds, subject_ids, window_size=5):
    N, num_aus = preds.shape
    smoothed_preds = np.zeros_like(preds)
    unique_subjects = np.unique(subject_ids)

    for sub_id in unique_subjects:
        idxs = np.where(subject_ids == sub_id)[0]
        if len(idxs) == 0: continue
        sub_pred = preds[idxs]
        for au in range(num_aus):
            kernel = np.ones(window_size) / window_size
            smoothed_au = np.convolve(sub_pred[:, au], kernel, mode='same')
            smoothed_preds[idxs, au] = smoothed_au

    return smoothed_preds


def valid(mode, epoch, data_loader, net, opts):
    global iter_num
    global best_valid_score_
    global best_valid_mae_

    net.eval()

    total_pred = []
    total_label = []
    total_ids = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img = data[0].to(device)
            ref_img = data[1].to(device)
            flow = data[2].to(device)
            label = data[3].to(device)
            subject_id = data[5]

            pred_mean, _, _, _, _, _ = net(img, ref_img, flow)

            total_pred.append(pred_mean.detach().cpu().numpy())
            total_label.append(label.detach().cpu().numpy())
            total_ids.append(subject_id.numpy())

    total_pred = np.concatenate(total_pred, axis=0)
    total_label = np.concatenate(total_label, axis=0)
    total_ids = np.concatenate(total_ids, axis=0)

    raw_MAE, raw_ICC, raw_F1, raw_MAE_plus = utils.evaluate_au(total_pred, total_label)

    # 平滑处理
    smooth_pred = perform_smoothing(total_pred, total_ids, window_size=5)
    smooth_pred = np.clip(smooth_pred, 0.0, 5.0)
    smt_MAE, smt_ICC, smt_F1, smt_MAE_plus = utils.evaluate_au(smooth_pred, total_label)

    print('==========================================================')
    print(f'{mode}_Iter: [{iter_num}][{epoch + 1}] (Mode: {opts.align_mode.upper()})')
    print(f'Raw      -> ICC: {raw_ICC.mean():.4f} | MAE: {raw_MAE.mean():.4f} | F1: {raw_F1.mean():.4f}')
    print(f'Smoothed -> ICC: {smt_ICC.mean():.4f} | MAE: {smt_MAE.mean():.4f} | F1: {smt_F1.mean():.4f}')
    print('==========================================================')

    current_score = raw_ICC.mean()

    if current_score > best_valid_score_:
        best_valid_score_ = current_score
        best_valid_mae_ = raw_MAE.mean()
        print(f'>>> New Best Model (Raw ICC: {best_valid_score_:.4f}) <<<')
        if opts.save_checkpoint:
            utils.save_checkpoint(opts.save_dir, iter_num, net)

    utils.save_loss(opts.save_dir, mode,
                    (iter_num, raw_ICC.mean(), raw_MAE.mean()) +
                    tuple(raw_ICC.tolist()) + tuple(raw_MAE.tolist()))

    net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AU_DISFA')
    # 建议修改 snapshot 名称
    parser.add_argument('--snapshot', type=str, default='R8_AffectNet_AUGuided')
    parser.add_argument('--train_num', type=int, default=1)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_valid', type=int, default=256)
    parser.add_argument('--lr_backbone', type=str, default='1e-5,1e-6')
    parser.add_argument('--lr_new', type=str, default='3e-4,3e-5')
    parser.add_argument('--lr_patch', type=str, default='1e-4,1e-5')
    parser.add_argument('--lr_attention', type=str, default='3e-4,3e-5')
    parser.add_argument('--use_Adam', type=int, default=1)
    parser.add_argument('--decay_backbone', type=str, default='6')
    parser.add_argument('--decay_new', type=str, default='10')
    parser.add_argument('--decay_patch', type=str, default='10')
    parser.add_argument('--decay_attention', type=str, default='10')
    parser.add_argument('--momentum', type=str, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=1363)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--last_epoch', type=int, default=3)

    parser.add_argument('--json_dir', type=str, default=r'D:\\RA-UWML-AU-Pytorch-master\\preprocess\\Disfa\\Dataset')
    parser.add_argument('--json_name', type=str, default='Disfa_Right_224_1.0.json')
    parser.add_argument('--use_sampler', type=int, default=0)
    parser.add_argument('--fold', type=str, default='CCNN')
    parser.add_argument('--save_dir', type=str,
                        default=r'D:\RA-UWML-AU-Pytorch-master\RA-UWML-AU-Pytorch-master+allimf\results')
    parser.add_argument('--save_checkpoint', type=int, default=1)

    parser.add_argument('--patch_number', type=int, default=3)
    parser.add_argument('--individual_featured', type=int, default=256)
    parser.add_argument('--backbone', type=str, default='imf')
    parser.add_argument('--target_task', type=str, default='regress,uncertain')
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--head_number', type=int, default=2)
    parser.add_argument('--init_sigma', type=float, default=-2.0)
    # 默认 Rank 权重设为 0.5 (Warm-up 后生效)
    parser.add_argument('--lambda_rank', type=float, default=0.5)

    parser.add_argument('--align_mode', type=str, default='implicit', choices=['none', 'flow', 'implicit'])

    # R8: ONLY experimental variable is backbone initialization.
    parser.add_argument('--affect_pretrained', type=str, required=True,
                        help='Path to official AffectNet ResNet50 CAM + AU-heatmap-guided model.pt')

    opts = parser.parse_args()

    main(opts)