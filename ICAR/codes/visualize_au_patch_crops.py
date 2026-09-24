import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import model

# 你的 AU 顺序
AU_LIST = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9",
           "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]


class SimpleOpts:
    pass


def build_model(device, patch_number=12, ckpt_path=None):
    opts = SimpleOpts()
    opts.backbone = "resnet18"
    opts.strides = "1,2,2,1"
    opts.pretrain_backbone = ""
    opts.patch_number = patch_number
    opts.individual_featured = 256
    opts.target_task = "regress,uncertain"
    opts.block_num = 2
    opts.head_number = 2

    net = model.AUResnet(opts).to(device)
    net.eval()

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "net_state_dict" in ckpt:
            net.load_state_dict(ckpt["net_state_dict"])
        else:
            net.load_state_dict(ckpt)

    return net


def load_image(image_path, device):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # 按你训练时的尺寸来
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])
    x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return x


def get_thetas(net, x):
    """只跑到 PatchGenerator，拿到每个 patch 的仿射矩阵 theta。"""
    with torch.no_grad():
        feat = net.backbone(x)                    # [1,512,h,w]
        xs = net.patch_proposal.localization(feat)
        xs = F.adaptive_avg_pool2d(xs, (1, 1))
        xs = torch.flatten(xs, 1)                 # [1,1024]
        theta = net.patch_proposal.fc_loc(xs)     # [1, 2*3*P]
        theta = theta.view(-1, net.patch_number, 2, 3)  # [1,P,2,3]
    return theta[0]  # [P,2,3]


def crop_patches(x, theta, out_size=(56, 56)):
    """
    用 grid_sample 根据 theta 从原图 x 裁出每个 patch。
    x: [1,3,H,W]
    theta: [P,2,3]
    返回: [P,3,h,w]
    """
    B, C, H, W = x.shape
    P = theta.shape[0]

    # 为每个 patch 生成一个 grid 并裁剪
    patches = []
    for p in range(P):
        theta_p = theta[p].unsqueeze(0)             # [1,2,3]
        grid = F.affine_grid(theta_p, torch.Size((1, C, *out_size)),
                             align_corners=True)    # [1,h,w,2]
        patch = F.grid_sample(x, grid, align_corners=True)  # [1,3,h,w]
        patches.append(patch[0])
    patches = torch.stack(patches, dim=0)           # [P,3,h,w]
    return patches


def visualize_patch_crops(image_path, ckpt_path=None,
                          save_path="vis_patch_crops.png",
                          device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    net = build_model(device, patch_number=12, ckpt_path=ckpt_path)

    x = load_image(image_path, device)  # [1,3,H,W]
    theta = get_thetas(net, x)          # [P,2,3]
    patches = crop_patches(x, theta, out_size=(56, 56))  # [P,3,56,56]

    patches_np = patches.detach().cpu().numpy()
    patches_np = (patches_np * 0.5 + 0.5).clip(0, 1)  # 反归一化到 [0,1]

    P, _, h, w = patches_np.shape
    rows, cols = 3, 4

    fig, axes = plt.subplots(rows, cols, figsize=(8, 6))
    for i in range(P):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = patches_np[i].transpose(1, 2, 0)    # [H,W,3]
        ax.imshow(img)
        au_name = AU_LIST[i] if i < len(AU_LIST) else f"patch{i}"
        ax.set_title(au_name, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"saved patch crops to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="要可视化的那张脸图路径")
    parser.add_argument("--ckpt", type=str, default="",
                        help="(可选) 训练好的 AUResnet 权重路径")
    parser.add_argument("--save", type=str, default="vis_patch_crops.png",
                        help="输出图片路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda 或 cpu")
    args = parser.parse_args()

    visualize_patch_crops(args.image,
                          ckpt_path=args.ckpt,
                          save_path=args.save,
                          device=args.device)


if __name__ == "__main__":
    main()
