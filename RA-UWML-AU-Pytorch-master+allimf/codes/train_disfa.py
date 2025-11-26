import random
import numpy as np
import torch
import os
import argparse
import json
import model
from params import ParamsControl
from dataset_disfa import get_data_loader
from loss import UncertainLoss
import utils
import torchvision.utils as vutils

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # 保证卷积算法确定性
    torch.backends.cudnn.benchmark = False    # 关闭自动优化，牺牲一点点速度换取稳定性
    print(f'[INFO] Random Seed set to: {seed}')

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

    # MODIFIED: Use AU_IMF_Net
    net = model.AU_IMF_Net(opts).to(device)

    params = ParamsControl(opts, net)
    criterion = UncertainLoss(opts.target_task, opts.init_sigma).to(device)

    AU_ID = ("AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17", "AU20", "AU25", "AU26")
    utils.save_loss(opts.save_dir, 'train', ("iter_num", "losses", "iccs", "maes"))
    utils.save_loss(opts.save_dir, 'valid', ("iter_num", "icc", "mae") + AU_ID * 2)

    for epoch in range(opts.start_epoch, opts.last_epoch):
        params.update(epoch)
        train(epoch, train_loader, valid_loader, net, criterion, params, opts)

    _print('valid best_valid_score:{}'.format(best_valid_score_))
    _print('valid best_valid_mae:{}'.format(best_valid_mae_))
    _print('finish...')
    _print('------' * 20)


def train(epoch, train_loader, valid_loader, net, criterion, params, opts):
    global iter_num

    losses = utils.AverageMeter()
    maes = utils.AverageMeter()
    iccs = utils.AverageMeter()

    net.train()

    for i, data in enumerate(train_loader):
        # MODIFIED: Unpack 4 elements
        img, ref_img, label, success = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        N, _, _, _ = img.shape

        # MODIFIED: Pass both images
        pred_mean, pred_std, pred_logits = net(img, ref_img)

        loss = criterion(pred_mean, pred_std, pred_logits, label, success)
        MAE, ICC, F1, MAE_plus = utils.evaluate_au(pred_mean, label)

        params.zero_grad()
        loss.backward()
        params.back_grad()

        losses.update(loss.item(), N)
        maes.update(MAE.mean(), N)
        iccs.update(ICC.mean(), N)
        # 你可以选择是否也记录 F1 的平均值，这里暂时只打印

        iter_num += 1
        if iter_num % opts.print_freq == 0:
            lrs = [group['lr'] for group in params.optimizers[0].param_groups]
            lr_str = f"LR_backbone={lrs[0]:.2e}" if len(lrs) > 0 else ""

            # --- 修改打印格式，加入 F1 和 MAE+ ---
            print('{0}_Iter: [{1}][{2}]\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'ICC {iccs.val:.3f} ({iccs.avg:.3f})\t'
                  'MAE {maes.val:.3f} ({maes.avg:.3f})\t'
                  'F1 {f1:.3f}\tMAE+ {mae_p:.3f}\t'  # 新增
                  '{lr}'.
                  format('Train', iter_num, epoch + 1, losses=losses, iccs=iccs, maes=maes,
                         f1=F1.mean(), mae_p=MAE_plus.mean(), lr=lr_str))

            # 打印预测分布统计
            pred_mean_stats = f"Pred[min={pred_mean.min():.3f}, max={pred_mean.max():.3f}, mean={pred_mean.mean():.3f}]"
            label_stats = f"Label[min={label.min():.3f}, max={label.max():.3f}, mean={label.mean():.3f}]"
            print(f"  {pred_mean_stats} | {label_stats}")

            utils.save_loss(opts.save_dir, 'train', (iter_num, losses.avg, iccs.avg, maes.avg))

        if iter_num % opts.valid_freq == 0:
            valid('valid', epoch, valid_loader, net, opts)


def valid(mode, epoch, data_loader, net, opts):
    global iter_num
    global best_valid_score_
    global best_valid_mae_

    net.eval()
    total_pred = []
    total_label = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img, ref_img, label, _ = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)

            # 这里的 debug save image 代码保持不变...

            pred_mean, _, _ = net(img, ref_img)
            total_pred += pred_mean.detach().cpu().tolist()
            total_label += label.detach().cpu().tolist()

    # --- 修改调用 ---
    total_MAE, total_ICC, total_F1, total_MAE_plus = utils.evaluate_au(total_pred, total_label)

    # 打印总览
    print('==========================================================')
    print('{0}_Iter: [{1}][{2}]\t'
          'ICC {icc:.4f}\t'
          'MAE {mae:.4f}\t'
          'F1 {f1:.4f}\t'
          'MAE+(Active) {mae_p:.4f}'.
          format(mode, iter_num, epoch + 1,
                 icc=total_ICC.mean(),
                 mae=total_MAE.mean(),
                 f1=total_F1.mean(),
                 mae_p=total_MAE_plus.mean()))

    AU_ID = ("AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17", "AU20", "AU25", "AU26")

    # 打印详细表格
    print(f"{'AU':<6} {'ICC':<8} {'F1':<8} {'MAE':<8} {'MAE+':<8}")
    print("-" * 40)
    for i, au_name in enumerate(AU_ID):
        print(f"{au_name:<6} {total_ICC[i]:<8.3f} {total_F1[i]:<8.3f} {total_MAE[i]:<8.3f} {total_MAE_plus[i]:<8.3f}")
    print('==========================================================')

    # 更新最佳模型逻辑 (通常还是以 ICC 为主，但你可以参考 F1)
    if total_ICC.mean() > best_valid_score_:
        best_valid_score_ = total_ICC.mean()
        best_valid_mae_ = total_MAE.mean()
        # 如果想以 F1 为准，可以改这里

    if opts.save_checkpoint:
        utils.save_checkpoint(opts.save_dir, iter_num, net)

    # 保存 Loss csv，这里列数变多了，需要把 save_loss 函数或者这里的 tuple 改一下
    # 建议只存核心指标，或者扩展 save_loss
    # 暂时保持原样或只存 ICC/MAE 均值，防止 CSV 格式错乱

    net.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AU_DISFA')
    parser.add_argument('--snapshot', type=str, default='DISFA_IMF_CCNN')
    parser.add_argument('--train_num', type=int, default=1)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_valid', type=int, default=1024)
    # 建议减小 backbone 的学习率，因为它是预训练的
    parser.add_argument('--lr_backbone', type=str, default='1e-5,1e-6')
    parser.add_argument('--lr_new', type=str, default='3e-4,3e-5')
    parser.add_argument('--lr_patch', type=str, default='0.0,0.0')
    parser.add_argument('--lr_attention', type=str, default='3e-4,3e-5')
    parser.add_argument('--use_Adam', type=int, default=1)
    parser.add_argument('--decay_backbone', type=str, default='10')
    parser.add_argument('--decay_new', type=str, default='10')
    parser.add_argument('--decay_patch', type=str, default='10')
    parser.add_argument('--decay_attention', type=str, default='10')
    parser.add_argument('--momentum', type=str, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=1363)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--last_epoch', type=int, default=15)

    # 路径配置
    parser.add_argument('--json_dir', type=str, default=r'D:\\RA-UWML-AU-Pytorch-master\\preprocess\\Disfa\\Dataset')
    parser.add_argument('--json_name', type=str, default='Disfa_Right_224_1.0.json')
    parser.add_argument('--use_sampler', type=int, default=0)
    parser.add_argument('--fold', type=str, default='CCNN')
    parser.add_argument('--save_dir', type=str, default=r'D:\RA-UWML-AU-Pytorch-master\RA-UWML-AU-Pytorch-master+allimf\results')
    parser.add_argument('--save_checkpoint', type=int, default=1)

    # 模型参数
    parser.add_argument('--patch_number', type=int, default=3)
    parser.add_argument('--individual_featured', type=int, default=256)
    parser.add_argument('--backbone', type=str, default='imf')  # Just a tag now
    parser.add_argument('--target_task', type=str, default='regress,uncertain')

    # IMF 特定参数
    #parser.add_argument('--imf_checkpoint', type=str, default='', help='Path to IMF pretrained .pth file')

    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--head_number', type=int, default=2)
    parser.add_argument('--init_sigma', type=float, default=-2.0)

    opts = parser.parse_args()

    main(opts)