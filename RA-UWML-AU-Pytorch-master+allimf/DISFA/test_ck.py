import os
import re
import glob
import csv
import argparse
from typing import List, Dict, Optional, Tuple
from types import SimpleNamespace

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

import model_R8_AffectAU as model


# ----------------------------
# 0) 兼容不同 forward 返回值
# ----------------------------

def forward_get_pred_mean(
    net: torch.nn.Module,
    x_cur: torch.Tensor,
    x_ref: torch.Tensor,
) -> torch.Tensor:
    """R8 forward: (current, reference, flow=None) -> 6 outputs."""
    out = net(x_cur, x_ref, None)
    if isinstance(out, (tuple, list)):
        pred_mean = out[0]
    else:
        pred_mean = out

    if pred_mean.ndim != 2 or pred_mean.shape[1] != 12:
        raise RuntimeError(
            f"Expected DISFA 12-AU regression output [N,12], got {tuple(pred_mean.shape)}"
        )
    return pred_mean


def load_disfa_checkpoint(net: torch.nn.Module, checkpoint: str, device: str) -> None:
    """Strictly load a full DISFA ICAR/R8 checkpoint saved by utils.save_checkpoint."""
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    except Exception:
        # Only for checkpoints you trust.
        state = torch.load(checkpoint, map_location=device, weights_only=False)

    if isinstance(state, dict) and "net_state_dict" in state:
        state = state["net_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(state)}")

    # Strip DataParallel prefix if necessary.
    if state and all(str(k).startswith("module.") for k in state.keys()):
        state = {str(k)[7:]: v for k, v in state.items()}

    incompatible = net.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.endswith("num_batches_tracked")]
    unexpected = list(incompatible.unexpected_keys)

    print("=" * 80)
    print("[CK+ R8] DISFA checkpoint:", checkpoint)
    print("[CK+ R8] missing keys:", len(missing))
    print("[CK+ R8] unexpected keys:", len(unexpected))
    if missing:
        print("[CK+ R8] missing examples:", missing[:20])
    if unexpected:
        print("[CK+ R8] unexpected examples:", unexpected[:20])

    if missing or unexpected:
        raise RuntimeError(
            "The DISFA checkpoint does not exactly match model_R8_AffectAU. "
            "Do not evaluate CK+ until missing/unexpected keys are zero."
        )

    print("[CK+ R8] PASS: full trained DISFA checkpoint loaded.")
    print("=" * 80)



# ----------------------------
# 1) 路径与帧排序
# ----------------------------
_FRAME_NUM_RE = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


def extract_frame_num(path: str) -> int:
    """从文件名末尾提取数字作为帧序号：frame_det_00_000001.bmp -> 1"""
    base = os.path.basename(path)
    m = _FRAME_NUM_RE.search(base)
    return int(m.group(1)) if m else 0


def find_aligned_dir(seq_dir: str) -> Optional[str]:
    """优先找 *_aligned 目录（如 001_aligned），找不到就返回 None"""
    cands = sorted([d for d in glob.glob(os.path.join(seq_dir, "*_aligned")) if os.path.isdir(d)])
    return cands[0] if len(cands) > 0 else None


def list_frames(seq_dir: str) -> List[str]:
    """
    适配你的结构：seq_dir/*_aligned/frame_det_*.bmp
    同时做多后缀兜底（bmp/png/jpg/jpeg）
    """
    aligned = find_aligned_dir(seq_dir)

    patterns: List[str] = []
    if aligned is not None:
        patterns.extend([
            os.path.join(aligned, "frame_det_*.bmp"),
            os.path.join(aligned, "frame_det_*.png"),
            os.path.join(aligned, "frame_det_*.jpg"),
            os.path.join(aligned, "frame_det_*.jpeg"),
        ])
    else:
        # 兜底：递归找 frame_det
        patterns.extend([
            os.path.join(seq_dir, "**", "frame_det_*.bmp"),
            os.path.join(seq_dir, "**", "frame_det_*.png"),
            os.path.join(seq_dir, "**", "frame_det_*.jpg"),
            os.path.join(seq_dir, "**", "frame_det_*.jpeg"),
        ])

    frames: List[str] = []
    for p in patterns:
        frames.extend(glob.glob(p, recursive="**" in p))

    # 去重 + 排序
    return sorted(set(frames), key=extract_frame_num)


# ----------------------------
# 2) 图像预处理（与训练一致）
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_img_rgb(path: str, tfm: T.Compose) -> torch.Tensor:
    """
    强制 convert("RGB")：
      - 避免灰度 bmp 读出来是 1 通道导致网络输入不匹配
    """
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)  # (1,3,H,W)


# ----------------------------
# 3) 推理：总强度曲线
# ----------------------------
@torch.no_grad()
def infer_total_intensity_curve(
    net: torch.nn.Module,
    frames: List[str],
    tfm: T.Compose,
    device: str,
    stride: int = 1,
    max_frames: int = 0,
) -> np.ndarray:
    """
    返回：
      intens: (T,)  每帧总强度（12 个 AU 回归值求和）
    """
    if len(frames) < 2:
        return np.array([], dtype=np.float64)

    used = frames[::max(1, stride)]
    if max_frames and len(used) > max_frames:
        used = used[:max_frames]

    if len(used) < 2:
        return np.array([], dtype=np.float64)

    # 参考帧：第一帧（中性）
    x_ref = load_img_rgb(used[0], tfm).to(device)

    intens: List[float] = []
    for fp in used:
        x_cur = load_img_rgb(fp, tfm).to(device)
        pred_mean = forward_get_pred_mean(net, x_cur, x_ref)  # (1,12)
        intens.append(float(pred_mean.sum(dim=1).item()))

    return np.asarray(intens, dtype=np.float64)


# ----------------------------
# 4) 论文指标（只保留 PCC / ICC / MAE）
# ----------------------------
def minmax_norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    return (x - mn) / (mx - mn + eps)


def gt_linear_01(T: int) -> np.ndarray:
    """论文 Eq.(9) 思路：强度真值按帧序线性从 0 到 1"""
    if T < 2:
        return np.zeros((T,), dtype=np.float64)
    return np.linspace(0.0, 1.0, num=T, dtype=np.float64)


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


def icc_paper(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    论文 ICC（按论文给出的 pooled mean/std 思路实现）：
      ICC = sum (xi - m_xy)(yi - m_xy) / ((n-1) * S_xy^2)

    这里的 S_xy^2 用 pooled mean 下的 pooled variance（把 x,y 视为两组观测）
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 2 or y.size != n:
        return 0.0

    mx = x.mean()
    my = y.mean()
    mxy = 0.5 * (mx + my)

    # pooled variance around pooled mean
    s2 = (((x - mxy) ** 2).sum() + ((y - mxy) ** 2).sum()) / (2 * max(1, (n - 1)) + eps)
    if s2 <= eps:
        return 0.0

    num = ((x - mxy) * (y - mxy)).sum()
    return float(num / ((n - 1) * s2 + eps))


def mae_paper(x: np.ndarray, y: np.ndarray) -> float:
    """论文 MAE: mean(|xi-yi|)"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size != x.size:
        return 0.0
    return float(np.abs(x - y).mean())


def summarize(arr: List[float]) -> Dict[str, float]:
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
    }


# ----------------------------
# 5) 主流程
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True, help=r"D:\CK+\Processed_224")
    parser.add_argument("--ckpt", type=str, required=True, help="DISFA 训练保存的权重 .pth")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--image_size", type=int, default=224)

    # 模型结构参数（保持和你当前工程一致）
    parser.add_argument("--patch_number", type=int, default=3)
    parser.add_argument("--individual_featured", type=int, default=256)
    parser.add_argument("--block_num", type=int, default=2)
    parser.add_argument("--head_number", type=int, default=2)
    parser.add_argument("--target_task", type=str, default="regress,uncertain",
                        help="Must match the DISFA R8 training architecture.")

    # 与训练保持一致（你的 Implicit_rank ckpt 需要 implicit 才能完整加载 alignment 权重）
    parser.add_argument("--align_mode", type=str, default="implicit", choices=["none", "implicit"],
                        help="对齐模式：none 或 implicit（flow 需要额外光流输入，当前脚本不支持）")


    # 评估采样
    parser.add_argument("--stride", type=int, default=1, help="每隔 stride 帧取一帧（加速）")
    parser.add_argument("--max_frames", type=int, default=0, help="每个序列最多用多少帧（0=不限制）")
    parser.add_argument("--max_seqs", type=int, default=0, help="最多跑多少个序列（0=不限制）")

    # 输出
    parser.add_argument("--out_csv", type=str, default="", help="保存每个序列的 PCC/ICC/MAE 到 CSV")

    args = parser.parse_args()

    # 1) Build exactly the same architecture used by the best DISFA R8 run.
    # Evaluation does NOT need to reload the external AffectNet model.pt:
    # the full trained DISFA checkpoint overwrites all model parameters/buffers.
    target_task = args.target_task.strip()
    opts_for_model = SimpleNamespace(
        patch_number=args.patch_number,
        individual_featured=args.individual_featured,
        block_num=args.block_num,
        head_number=args.head_number,
        target_task=target_task,
        align_mode=args.align_mode,
        affect_pretrained=None,
    )

    print("[CK+ R8] model config:")
    print("  patch_number       =", opts_for_model.patch_number)
    print("  individual_featured=", opts_for_model.individual_featured)
    print("  block_num          =", opts_for_model.block_num)
    print("  head_number        =", opts_for_model.head_number)
    print("  target_task        =", opts_for_model.target_task)
    print("  align_mode         =", opts_for_model.align_mode)

    net = model.AU_IMF_Net(opts_for_model).to(args.device)
    load_disfa_checkpoint(net, args.ckpt, args.device)
    net.eval()

    tfm = build_transform(args.image_size)

    # 2) 序列目录
    all_dirs = [os.path.join(args.processed_dir, d) for d in sorted(os.listdir(args.processed_dir))]
    seq_dirs = [d for d in all_dirs if os.path.isdir(d)]
    if args.max_seqs and len(seq_dirs) > args.max_seqs:
        seq_dirs = seq_dirs[:args.max_seqs]

    print(f"[INFO] found seq dirs: {len(seq_dirs)}")

    # 3) CSV
    csv_f = None
    writer = None
    if args.out_csv:
        csv_f = open(args.out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_f)
        writer.writerow(["seq_name", "T", "PCC", "ICC", "MAE"])

    # 4) 遍历评估（只算论文三指标）
    all_pcc: List[float] = []
    all_icc: List[float] = []
    all_mae: List[float] = []

    used = 0
    for idx, sd in enumerate(seq_dirs, start=1):
        seq_name = os.path.basename(sd)
        frames = list_frames(sd)

        if len(frames) < 2:
            print(f"[SKIP] {seq_name}: not enough frames")
            continue

        intens = infer_total_intensity_curve(
            net=net,
            frames=frames,
            tfm=tfm,
            device=args.device,
            stride=args.stride,
            max_frames=args.max_frames,
        )
        if intens.size < 2:
            print(f"[SKIP] {seq_name}: inference failed/empty")
            continue

        # 当前论文稿件的 CK+ cross-dataset setting：序列预测归一化到[0,1]；GT 用线性插值 0->1
        pred_y = minmax_norm(intens)
        gt_y = gt_linear_01(len(intens))

        pcc = pcc_paper(pred_y, gt_y)
        icc = icc_paper(pred_y, gt_y)
        mae = mae_paper(pred_y, gt_y)

        used += 1
        all_pcc.append(pcc)
        all_icc.append(icc)
        all_mae.append(mae)

        print(f"[{idx:4d}/{len(seq_dirs)}] {seq_name}  T={len(intens):3d} | PCC={pcc:.4f} | ICC={icc:.4f} | MAE={mae:.4f}")

        if writer is not None:
            writer.writerow([seq_name, len(intens), f"{pcc:.6f}", f"{icc:.6f}", f"{mae:.6f}"])

    if csv_f is not None:
        csv_f.close()
        print(f"[INFO] saved csv -> {args.out_csv}")

    # 5) 汇总
    s_pcc = summarize(all_pcc)
    s_icc = summarize(all_icc)
    s_mae = summarize(all_mae)

    print("\n================== SUMMARY ==================")
    print(f"Seq used: {used} / {len(seq_dirs)}")
    print(f"PCC : mean={s_pcc['mean']:.4f}, median={s_pcc['median']:.4f}, p25={s_pcc['p25']:.4f}, p75={s_pcc['p75']:.4f}")
    print(f"ICC : mean={s_icc['mean']:.4f}, median={s_icc['median']:.4f}, p25={s_icc['p25']:.4f}, p75={s_icc['p75']:.4f}")
    print(f"MAE : mean={s_mae['mean']:.4f}, median={s_mae['median']:.4f}, p25={s_mae['p25']:.4f}, p75={s_mae['p75']:.4f}")
    print("=============================================\n")


if __name__ == "__main__":
    main()
