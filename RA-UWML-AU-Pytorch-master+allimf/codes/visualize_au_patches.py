import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 引入你的模型文件
import model

# 自定义补丁名称
PATCH_NAMES = ["Patch1: Upper/Brows", "Patch2: Middle/Eyes", "Patch3: Lower/Mouth", "Patch4: Glabella(AU4)"]
COLORS = ["lime", "cyan", "yellow", "red"]  # 第4个用红色高亮


class SimpleOpts:
    """构造 AU_IMF_Net 需要的参数配置"""

    def __init__(self, patch_number=4):
        self.patch_number = patch_number
        self.individual_featured = 256
        self.target_task = "regress,uncertain"
        self.block_num = 2
        self.head_number = 2
        self.backbone = "imf"
        # 下面这些参数在模型初始化时可能用不到，但为了防止报错加上
        self.snapshot = "debug"


def build_model(device, patch_number=4, ckpt_path=None):
    opts = SimpleOpts(patch_number=patch_number)

    # 初始化模型
    net = model.AU_IMF_Net(opts).to(device)
    net.eval()

    # 加载权重 (如果有)
    if ckpt_path is not None and len(ckpt_path) > 0 and os.path.isfile(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "net_state_dict" in ckpt:
            net.load_state_dict(ckpt["net_state_dict"])
        else:
            net.load_state_dict(ckpt)
    else:
        print("No checkpoint loaded. Visualizing INITIALIZATION (Hard-coded positions).")

    return net


def load_image(image_path, device):
    """读入图片并预处理"""
    img_pil = Image.open(image_path).convert("RGB")
    W, H = img_pil.size

    # 保持和训练一致的 Resize (通常 DISFA 预处理是 224 或者 256，这里假设 224)
    # 注意：可视化的原图不需要 Resize，但送入网络的 Tensor 需要
    process = transforms.Compose([
        transforms.Resize((224, 224)),  # 这里要和训练时的输入尺寸一致，否则 patch 比例不对
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    x = process(img_pil).unsqueeze(0).to(device)  # [1,3,224,224]
    return x, img_pil


def theta_to_bbox(theta, H, W):
    """
    根据 theta [sx, 0, tx, 0, sy, ty] 计算 bbox
    注意：theta 是 affine matrix 的简化参数
    Affine Grid 公式: x_s = theta[0,0]*x_t + theta[0,2]
    """
    # 提取缩放和平移
    # theta shape: [2, 3]
    # [[sx, 0, tx],
    #  [0, sy, ty]]

    sx = theta[0, 0]
    sy = theta[1, 1]
    tx = theta[0, 2]
    ty = theta[1, 2]

    # 在标准化坐标系 [-1, 1] 中的中心点是 (tx, ty)
    # 宽度覆盖范围是 2 * sx, 高度覆盖范围是 2 * sy
    # (这是基于 affine_grid 的逻辑，如果 sx=1，则覆盖整个 [-1,1] 也就是全图)

    # 计算边界 (Normalized)
    # grid sample 的采样范围是 [-1, 1]
    # Box 的左边界: center_x - scale_x
    x_min_n = tx - sx
    x_max_n = tx + sx
    y_min_n = ty - sy
    y_max_n = ty + sy

    # 映射回像素坐标
    # [-1, 1] -> [0, W]
    def to_pixel(val, size):
        return (val + 1) / 2 * size

    x_min = to_pixel(x_min_n, W)
    x_max = to_pixel(x_max_n, W)
    y_min = to_pixel(y_min_n, H)
    y_max = to_pixel(y_max_n, H)

    return x_min, y_min, x_max, y_max


def visualize_patches(image_path, ckpt_path=None, save_path="vis_patches.png", device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. 构建模型 (Patch Number 必须设为 4)
    net = build_model(device, patch_number=3, ckpt_path=ckpt_path)

    # 2. 准备数据
    x_curr, img_pil = load_image(image_path, device)

    # 为了触发 patch_proposal，我们需要 reference image。
    # 这里的目的是验证 Bias 初始化，所以我们即便传入一样的图，
    # Motion Feature 会接近 0，网络主要靠 fc_loc 的 bias (硬编码初始值) 输出结果。
    # 这样正好能验证你改的初始化对不对。
    x_ref = x_curr.clone()

    # 3. 前向传播 (模拟 AU_IMF_Net 的流程)
    with torch.no_grad():
        # Encoder
        f_c = net.encoder(x_curr)
        f_r = net.encoder(x_ref)

        # Alignment & Motion
        aligned_ref = net.alignment(f_c, f_r)
        motion_feature = f_c - aligned_ref

        # Patch Proposal
        # 这一步内部调用了 localization -> pooling -> fc_loc
        # 但 net.patch_proposal(x) 返回的是 grid_sample 后的图
        # 我们需要中间变量 theta，所以手动跑一下内部逻辑

        feat = net.patch_proposal.localization(motion_feature)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)
        theta = net.patch_proposal.fc_loc(feat)  # [1, 2*3*P]
        theta = theta.view(-1, net.patch_number, 2, 3)  # [1, P, 2, 3]

    theta_np = theta[0].cpu().numpy()  # [P, 2, 3]

    # 4. 绘图
    plt.figure(figsize=(8, 8))
    plt.imshow(img_pil)
    ax = plt.gca()

    W_disp, H_disp = img_pil.size

    print("--- Patch Coordinates (Normalized [-1, 1]) ---")
    for i in range(net.patch_number):
        # 转换坐标
        x_min, y_min, x_max, y_max = theta_to_bbox(theta_np[i], H_disp, W_disp)

        w = x_max - x_min
        h = y_max - y_min

        color = COLORS[i] if i < len(COLORS) else "white"
        name = PATCH_NAMES[i] if i < len(PATCH_NAMES) else f"Patch {i + 1}"

        # 打印调试信息
        print(
            f"{name}: Center=({theta_np[i, 0, 2]:.2f}, {theta_np[i, 1, 2]:.2f}), Scale=({theta_np[i, 0, 0]:.2f}, {theta_np[i, 1, 1]:.2f})")

        # 画框
        rect = patches.Rectangle((x_min, y_min), w, h,
                                 linewidth=2,
                                 edgecolor=color,
                                 facecolor="none",
                                 alpha=0.9)
        ax.add_patch(rect)

        # 标签
        ax.text(x_min, max(y_min - 5, 0),
                name,
                fontsize=9,
                color="white",
                weight="bold",
                bbox=dict(facecolor=color, alpha=0.6, pad=1.5, edgecolor='none'))

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")
    # plt.show() # 如果在服务器上跑，请注释掉这行


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path (optional)")
    parser.add_argument("--save", type=str, default="check_patches.png", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    visualize_patches(args.image,
                      ckpt_path=args.ckpt,
                      save_path=args.save,
                      device=args.device)


if __name__ == "__main__":
    main()