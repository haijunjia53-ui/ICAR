#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# 解决 Windows 下可能出现的 OpenMP/DLL 冲突，必须放在 import torch 之前
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import model_bp4d as model
from params import ParamsControl
from dataset_bp4d import get_data_loader
from loss_bp4d import UncertainLoss, L2HingeRankLoss
import utils


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random Seed set to: {seed}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iter_num = 0

best_valid_score_ = 0.0
best_valid_mae_ = 100000.0


def load_frame_checkpoint(
    net,
    checkpoint_path,
):
    if not checkpoint_path:
        print(
            "[Temporal] No frame checkpoint provided."
        )
        return

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"frame checkpoint 不存在："
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "net_state_dict" in checkpoint:
        state_dict = checkpoint[
            "net_state_dict"
        ]
    else:
        state_dict = checkpoint

    # 兼容 DataParallel 保存出的 module.xxx
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    incompatible = net.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    missing_keys = list(
        incompatible.missing_keys
    )
    unexpected_keys = list(
        incompatible.unexpected_keys
    )

    print(
        "[Temporal] Missing keys:",
        missing_keys,
    )
    print(
        "[Temporal] Unexpected keys:",
        unexpected_keys,
    )

    illegal_missing = [
        key
        for key in missing_keys
        if not key.startswith("temporal_head.")
    ]

    if illegal_missing:
        raise RuntimeError(
            "除了 temporal_head 以外还有参数没有加载："
            f"{illegal_missing}"
        )

    if unexpected_keys:
        print(
            "[Warning] checkpoint 中存在未使用参数：",
            unexpected_keys,
        )

    print(
        f"[Temporal] Loaded frame checkpoint: "
        f"{checkpoint_path}"
    )


def configure_temporal_stage(
    net,
    stage,
):
    """
    stage 1:
        只训练 temporal_head。

    stage 2:
        temporal_head + new heads + self_attention。

    stage 3:
        stage 2 +
        alignment +
        encoder.layer3。

    patch proposal 默认冻结，避免时序实验同时改变 patch 定位。
    """
    stage = int(stage)

    if stage not in (1, 2, 3):
        raise ValueError(
            f"temporal_stage 必须为 1/2/3，"
            f"当前为 {stage}"
        )

    for parameter in net.parameters():
        parameter.requires_grad = False

    for parameter in net.temporal_head.parameters():
        parameter.requires_grad = True

    if stage >= 2:
        for parameter in net.new.parameters():
            parameter.requires_grad = True

        for parameter in net.self_attention.parameters():
            parameter.requires_grad = True

    if stage >= 3:
        for parameter in net.alignment.parameters():
            parameter.requires_grad = True

        for parameter in net.encoder.layer3.parameters():
            parameter.requires_grad = True

    trainable = sum(
        parameter.numel()
        for parameter in net.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in net.parameters()
    )

    print(
        f"[Temporal Stage] stage={stage}, "
        f"trainable={trainable:,}/{total:,} "
        f"({100.0 * trainable / total:.2f}%)"
    )
def set_module_eval(module):
    module.eval()


def configure_train_modes(
    net,
    stage,
):
    """
    先 net.train()，再精确设置冻结模块的 train/eval 状态。
    """
    net.train()

    stage = int(stage)

    if stage == 1:
        net.encoder.eval()
        net.alignment.eval()
        net.patch_proposal.eval()
        net.self_attention.eval()
        net.new.eval()

        net.temporal_head.train()

    elif stage == 2:
        net.encoder.eval()
        net.alignment.eval()
        net.patch_proposal.eval()

        net.self_attention.train()
        net.new.train()
        net.temporal_head.train()

    elif stage == 3:
        # encoder 的 layer0/1/2 仍冻结，
        # 因此这些部分保持 eval。
        net.encoder.eval()

        # 只让 layer3 进入 train。
        net.encoder.layer3.train()

        net.alignment.train()
        net.patch_proposal.eval()
        net.self_attention.train()
        net.new.train()
        net.temporal_head.train()

def main(opts):
    global best_valid_score_
    global best_valid_mae_

    setup_seed(opts.seed)

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    opts.save_dir = os.path.join(opts.save_dir, opts.snapshot)
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    logging = utils.init_log(os.path.join(opts.save_dir, f"{opts.snapshot}.log"))
    _print = logging.info
    _print(opts.snapshot)

    opts.train_num = "{:0>4}".format(opts.train_num)
    opts.save_dir = os.path.join(opts.save_dir, opts.train_num)
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    _print(opts.train_num)

    # Dataset 加载
    train_loader, n_data_train = get_data_loader(opts, "train", opts.fold)
    valid_loader, n_data_valid = get_data_loader(opts, "valid", opts.fold)

    # 从 BP4D json 读取 AU 数量和 AU 名称，写回 opts 给 model_bp4d 使用
    opts.au_number = int(train_loader.dataset.au_number)
    opts.au_ids = list(train_loader.dataset.au_ids)

    # 如果没有手动设置 valid_freq，则每个 epoch 验证一次
    if opts.valid_freq <= 0:
        opts.valid_freq = max(1, len(train_loader))

    _print("Train Dataset Info")
    _print(f"train total samples:  {n_data_train}")
    _print(f"train batch_size:     {opts.batch_size_train}")
    _print(f"train batch_num:      {len(train_loader)}")
    _print(f"total {len(train_loader)}*{opts.last_epoch} = {len(train_loader) * opts.last_epoch} iters")

    _print("Valid Dataset Info")
    _print(f"valid total samples:  {n_data_valid}")
    _print(f"valid batch_size:     {opts.batch_size_valid}")
    _print(f"valid batch_num:      {len(valid_loader)}")
    _print(f"total {len(valid_loader)}*{opts.last_epoch} = {len(valid_loader) * opts.last_epoch} iters")
    _print(f"BP4D AU IDs: {opts.au_ids}")
    _print(f"BP4D AU number: {opts.au_number}")
    _print(f"patch_au_num: {opts.patch_au_num}")

    with open(os.path.join(opts.save_dir, "opts_setting.txt"), "w", encoding="utf-8") as f:
        json.dump(opts.__dict__, f, indent=2, ensure_ascii=False)

    net = model.AU_IMF_Net(opts).to(device)

    # 1. 加载原单帧最佳 checkpoint
    load_frame_checkpoint(
        net,
        opts.frame_checkpoint,
    )

    # 2. 根据阶段冻结或解冻参数
    configure_temporal_stage(
        net,
        opts.temporal_stage,
    )

    # 3. 必须在冻结设置完成后再创建 optimizer
    params = ParamsControl(
        opts,
        net,
    )

    criterion = UncertainLoss(
        opts.target_task,
        opts.init_sigma,
    ).to(device)

    rank_criterion = L2HingeRankLoss(
        margin_base=0.5,
        margin_slope=0.2,
    ).to(device)

    AU_ID = tuple(opts.au_ids)
    utils.save_loss(opts.save_dir, "train", ("iter_num", "losses", "iccs", "maes"))
    utils.save_loss(opts.save_dir, "valid", ("iter_num", "icc", "mae") + AU_ID * 2)

    for epoch in range(opts.start_epoch, opts.last_epoch):
        params.update(epoch)
        train(epoch, train_loader, valid_loader, net, criterion, rank_criterion, params, opts)

    _print(f"valid best_valid_score:{best_valid_score_}")
    _print(f"valid best_valid_mae:{best_valid_mae_}")
    _print("finish...")
    _print("------" * 20)


def train(
    epoch,
    train_loader,
    valid_loader,
    net,
    criterion,
    rank_criterion,
    params,
    opts,
):
    global iter_num

    losses = utils.AverageMeter()
    maes = utils.AverageMeter()
    iccs = utils.AverageMeter()
    align_losses = utils.AverageMeter()

    configure_train_modes(
        net,
        opts.temporal_stage,
    )

    if epoch < opts.rank_warmup_epoch:
        current_lambda_rank = 0.0
    else:
        current_lambda_rank = opts.lambda_rank

    accum_steps = max(
        1,
        int(opts.accum_steps),
    )

    params.zero_grad()

    for i, data in enumerate(train_loader):
        img = data[0].to(
            device,
            non_blocking=True,
        )
        ref_img = data[1].to(
            device,
            non_blocking=True,
        )
        flow = data[2].to(
            device,
            non_blocking=True,
        )
        label = data[3].to(
            device,
            non_blocking=True,
        )
        success = data[4].to(
            device,
            non_blocking=True,
        )
        subject_id = data[5].to(
            device,
            non_blocking=True,
        )

        N = img.shape[0]

        (
            pred_mean,
            pred_std,
            pred_logits,
            est_flow,
            aligned_ref,
            f_c,
        ) = net.forward_sequence(
            img,
            ref_img,
            flow,
        )

        loss_main = criterion(
            pred_mean,
            pred_std,
            pred_logits,
            label,
            success,
        )

        if current_lambda_rank > 0:
            loss_rank = rank_criterion(
                pred_mean,
                label,
                success,
                groups=subject_id,
            )
        else:
            loss_rank = torch.tensor(
                0.0,
                device=device,
            )

        loss_align = torch.tensor(
            0.0,
            device=device,
        )

        # stage 1/2 中 alignment 冻结，
        # 没必要计算 alignment loss。
        if (
            opts.align_mode == "implicit"
            and int(opts.temporal_stage) >= 3
        ):
            loss_align_mse = F.mse_loss(
                aligned_ref,
                f_c.detach(),
            )

            if est_flow is not None:
                dy = torch.abs(
                    est_flow[:, :, 1:, :]
                    - est_flow[:, :, :-1, :]
                ).mean()

                dx = torch.abs(
                    est_flow[:, :, :, 1:]
                    - est_flow[:, :, :, :-1]
                ).mean()

                loss_smooth = dx + dy
            else:
                loss_smooth = torch.tensor(
                    0.0,
                    device=device,
                )

            loss_align = (
                opts.lambda_align_mse
                * loss_align_mse
                + opts.lambda_align_smooth
                * loss_smooth
            )

        full_loss = (
            loss_main
            + current_lambda_rank * loss_rank
            + loss_align
        )

        # 梯度累积只缩放 backward 使用的 loss。
        backward_loss = (
            full_loss / accum_steps
        )

        backward_loss.backward()

        should_step = (
            (i + 1) % accum_steps == 0
            or (i + 1) == len(train_loader)
        )

        if should_step:
            if opts.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for p in net.parameters()
                        if p.requires_grad
                        and p.grad is not None
                    ],
                    max_norm=opts.grad_clip,
                )

            params.back_grad()
            params.zero_grad()

        MAE, ICC, F1, MAE_plus = (
            utils.evaluate_au(
                pred_mean,
                label,
            )
        )

        losses.update(
            full_loss.item(),
            N,
        )
        maes.update(
            MAE.mean(),
            N,
        )
        iccs.update(
            ICC.mean(),
            N,
        )
        align_losses.update(
            loss_align.item(),
            N,
        )

        iter_num += 1

        if iter_num % opts.print_freq == 0:
            lr_dict = params.get_lrs()

            print(
                f"Train_Iter: [{iter_num}]"
                f"[Epoch {epoch + 1}] "
                f"Loss {losses.val:.3f} "
                f"(Reg:{loss_main.item():.3f}, "
                f"Rank({current_lambda_rank}):"
                f"{loss_rank.item():.3f}, "
                f"Align:{loss_align.item():.4f}) "
                f"ICC {iccs.val:.3f} "
                f"({iccs.avg:.3f}) "
                f"MAE {maes.val:.3f} "
                f"({maes.avg:.3f}) "
                f"LR_temporal="
                f"{lr_dict['temporal']:.2e}"
            )

            utils.save_loss(
                opts.save_dir,
                "train",
                (
                    iter_num,
                    losses.avg,
                    iccs.avg,
                    maes.avg,
                ),
            )

        if iter_num % opts.valid_freq == 0:
            valid(
                "valid",
                epoch,
                valid_loader,
                net,
                opts,
            )


def perform_smoothing(preds, subject_ids, window_size=5):
    N, num_aus = preds.shape
    smoothed_preds = np.zeros_like(preds)
    unique_subjects = np.unique(subject_ids)

    for sub_id in unique_subjects:
        idxs = np.where(subject_ids == sub_id)[0]
        if len(idxs) == 0:
            continue

        sub_pred = preds[idxs]
        for au in range(num_aus):
            kernel = np.ones(window_size) / window_size
            smoothed_au = np.convolve(sub_pred[:, au], kernel, mode="same")
            smoothed_preds[idxs, au] = smoothed_au

    return smoothed_preds


def valid(
    mode,
    epoch,
    data_loader,
    net,
    opts,
):
    global iter_num
    global best_valid_score_
    global best_valid_mae_

    net.eval()

    total_pred = []
    total_label = []
    total_subject_ids = []
    total_sequence_ids = []
    total_center_positions = []
    total_global_indices = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img = data[0].to(
                device,
                non_blocking=True,
            )
            ref_img = data[1].to(
                device,
                non_blocking=True,
            )
            flow = data[2].to(
                device,
                non_blocking=True,
            )
            label = data[3].to(
                device,
                non_blocking=True,
            )

            subject_id = data[5]
            sequence_id = data[6]
            center_pos = data[7]
            global_idx = data[8]

            (
                pred_mean,
                _,
                _,
                _,
                _,
                _,
            ) = net.forward_sequence(
                img,
                ref_img,
                flow,
            )

            total_pred.append(
                pred_mean.detach().cpu().numpy()
            )
            total_label.append(
                label.detach().cpu().numpy()
            )
            total_subject_ids.append(
                subject_id.numpy()
            )
            total_sequence_ids.append(
                sequence_id.numpy()
            )
            total_center_positions.append(
                center_pos.numpy()
            )
            total_global_indices.append(
                global_idx.numpy()
            )

    total_pred = np.concatenate(
        total_pred,
        axis=0,
    )
    total_label = np.concatenate(
        total_label,
        axis=0,
    )
    total_subject_ids = np.concatenate(
        total_subject_ids,
        axis=0,
    )
    total_sequence_ids = np.concatenate(
        total_sequence_ids,
        axis=0,
    )
    total_center_positions = np.concatenate(
        total_center_positions,
        axis=0,
    )
    total_global_indices = np.concatenate(
        total_global_indices,
        axis=0,
    )

    # 按 global frame index 恢复原始顺序
    order = np.argsort(
        total_global_indices
    )

    total_pred = total_pred[order]
    total_label = total_label[order]
    total_subject_ids = total_subject_ids[order]
    total_sequence_ids = total_sequence_ids[order]
    total_center_positions = (
        total_center_positions[order]
    )
    total_global_indices = (
        total_global_indices[order]
    )

    # 确认验证 stride=1 时没有重复中心帧
    unique_count = len(
        np.unique(total_global_indices)
    )

    if unique_count != len(
        total_global_indices
    ):
        raise RuntimeError(
            f"验证中心帧存在重复："
            f"total={len(total_global_indices)}, "
            f"unique={unique_count}"
        )

    raw_MAE, raw_ICC, raw_F1, raw_MAE_plus = (
        utils.evaluate_au(
            total_pred,
            total_label,
        )
    )

    print(
        "=========================================================="
    )
    print(
        f"{mode}_Iter: [{iter_num}]"
        f"[{epoch + 1}] "
        f"(Temporal BiGRU, "
        f"T={opts.temporal_window}, "
        f"stage={opts.temporal_stage})"
    )
    print(
        f"Raw -> "
        f"ICC: {raw_ICC.mean():.4f} | "
        f"MAE: {raw_MAE.mean():.4f} | "
        f"F1: {raw_F1.mean():.4f}"
    )

    print(
        "Per-AU Raw ICC:",
        dict(
            zip(
                opts.au_ids,
                raw_ICC.tolist(),
            )
        ),
    )
    print(
        "Per-AU Raw MAE:",
        dict(
            zip(
                opts.au_ids,
                raw_MAE.tolist(),
            )
        ),
    )

    pred_mean_stat = total_pred.mean(
        axis=0
    )
    label_mean_stat = total_label.mean(
        axis=0
    )
    pred_std_stat = total_pred.std(
        axis=0
    )
    label_std_stat = total_label.std(
        axis=0
    )
    bias = (
        pred_mean_stat
        - label_mean_stat
    )

    print(
        "Per-AU F1:",
        dict(
            zip(
                opts.au_ids,
                raw_F1.tolist(),
            )
        ),
    )
    print(
        "Per-AU MAE+:",
        dict(
            zip(
                opts.au_ids,
                raw_MAE_plus.tolist(),
            )
        ),
    )
    print(
        "Per-AU Pred Mean:",
        dict(
            zip(
                opts.au_ids,
                pred_mean_stat.tolist(),
            )
        ),
    )
    print(
        "Per-AU Label Mean:",
        dict(
            zip(
                opts.au_ids,
                label_mean_stat.tolist(),
            )
        ),
    )
    print(
        "Per-AU Pred Std:",
        dict(
            zip(
                opts.au_ids,
                pred_std_stat.tolist(),
            )
        ),
    )
    print(
        "Per-AU Label Std:",
        dict(
            zip(
                opts.au_ids,
                label_std_stat.tolist(),
            )
        ),
    )
    print(
        "Per-AU Bias:",
        dict(
            zip(
                opts.au_ids,
                bias.tolist(),
            )
        ),
    )
    print(
        f"Validation center frames: "
        f"{len(total_global_indices)}"
    )
    print(
        "=========================================================="
    )

    current_score = float(
        raw_ICC.mean()
    )

    if current_score > best_valid_score_:
        best_valid_score_ = current_score
        best_valid_mae_ = float(
            raw_MAE.mean()
        )

        print(
            f">>> New Best Temporal Model "
            f"(Raw ICC: "
            f"{best_valid_score_:.4f}) <<<"
        )

        if opts.save_checkpoint:
            utils.save_checkpoint(
                opts.save_dir,
                iter_num,
                net,
            )

    utils.save_loss(
        opts.save_dir,
        mode,
        (
            iter_num,
            raw_ICC.mean(),
            raw_MAE.mean(),
        )
        + tuple(raw_ICC.tolist())
        + tuple(raw_MAE.tolist()),
    )

    # 恢复训练模式由 train() 下一轮重新精确设置。


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AU_BP4D")

    parser.add_argument("--snapshot", type=str, default="BP4D_Implicit")
    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_valid", type=int, default=512)

    parser.add_argument("--lr_backbone", type=str, default="1e-5,1e-6")
    parser.add_argument("--lr_new", type=str, default="3e-4,3e-5")
    parser.add_argument("--lr_patch", type=str, default="0.0,0.0")
    parser.add_argument("--lr_attention", type=str, default="3e-4,3e-5")

    parser.add_argument("--use_Adam", type=int, default=1)
    parser.add_argument("--decay_backbone", type=str, default="10")
    parser.add_argument("--decay_new", type=str, default="10")
    parser.add_argument("--decay_patch", type=str, default="10")
    parser.add_argument("--decay_attention", type=str, default="10")

    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument(
        "--valid_freq",
        type=int,
        default=-1,
        help="<=0 表示每个 epoch 验证一次。",
    )
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--last_epoch", type=int, default=15)

    parser.add_argument("--json_dir", type=str, default=r"E:\BP4D\Dataset")
    parser.add_argument("--json_name", type=str, default="BP4D_224.json")

    parser.add_argument("--use_sampler", type=int, default=0)
    parser.add_argument("--fold", type=str, default="fold1")
    parser.add_argument("--fold_num", type=int, default=3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--valid_subjects",
        type=str,
        default=None,
        help="手动指定验证受试者，例如 F001,F002,M005；指定后会覆盖 fold。",
    )
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    parser.add_argument("--neutral_ratio", type=float, default=2.0)
    parser.add_argument(
        "--filter_success",
        type=int,
        default=0,
        help="1 表示训练/验证只保留 OpenFace success=1 的帧。",
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="imagenet",
        choices=["imagenet", "dataset"],
        help="imagenet 更适合 ResNet 预训练；dataset 使用生成 json 里的 mean/std。",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"E:\BP4D\results",
    )
    parser.add_argument("--save_checkpoint", type=int, default=1)

    parser.add_argument("--patch_number", type=int, default=3)
    parser.add_argument("--individual_featured", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="imf")
    parser.add_argument("--target_task", type=str, default="regress,uncertain")
    parser.add_argument("--block_num", type=int, default=2)
    parser.add_argument("--head_number", type=int, default=2)
    parser.add_argument("--init_sigma", type=float, default=-2.0)

    # BP4D 5 个 AU：[AU06, AU10, AU12, AU14, AU17]
    # patch_au_num 总和必须等于 au_number。
    # 这里默认 [1,1,3]：AU06 / AU10 / AU12+AU14+AU17。
    parser.add_argument("--au_number", type=int, default=5)
    parser.add_argument("--patch_au_num", type=str, default="1,1,3")

    parser.add_argument("--lambda_rank", type=float, default=0.5)
    parser.add_argument("--rank_warmup_epoch", type=int, default=1)

    parser.add_argument("--lambda_align_mse", type=float, default=10.0)
    parser.add_argument("--lambda_align_smooth", type=float, default=0.1)

    parser.add_argument("--align_mode", type=str, default="implicit", choices=["none", "flow", "implicit"])
    parser.add_argument("--smooth_window", type=int, default=5)

    parser.add_argument(
        "--temporal_window",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--temporal_stride_train",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--temporal_stride_valid",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--temporal_hidden",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--temporal_dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--temporal_delta_scale",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--temporal_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
    )

    parser.add_argument(
        "--frame_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lr_alignment",
        type=str,
        default="0,0",
    )

    parser.add_argument(
        "--lr_temporal",
        type=str,
        default="3e-4,1e-4",
    )

    parser.add_argument(
        "--decay_alignment",
        type=str,
        default="6",
    )

    parser.add_argument(
        "--decay_temporal",
        type=str,
        default="6",
    )

    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    opts = parser.parse_args()
    main(opts)
