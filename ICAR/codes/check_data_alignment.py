import os
import cv2
import torch
import numpy as np
import argparse
from dataset_disfa import DISFADataset

# 配置颜色和字体用于图片标注
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 255, 0)  # Green
TEXT_SCALE = 0.8
THICKNESS = 2


def parse_args():
    parser = argparse.ArgumentParser(description='Check DISFA Dataset Reference Frames')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing the json config')
    parser.add_argument('--json_name', type=str, default='disfa_config.json', help='Name of the json file')
    parser.add_argument('--output_dir', type=str, default='./vis_check_result', help='Output directory for images')
    parser.add_argument('--split', type=str, default='CCNN', help='Split method: CCNN, fold1, etc.')
    return parser.parse_args()


class MockOpts:
    """模拟训练代码中的 opts 参数类"""

    def __init__(self, args):
        self.json_dir = args.json_dir
        self.json_name = args.json_name
        self.snapshot = 'normal'  # 非 debug 模式，使用 ImageNet 归一化
        self.batch_size_train = 1
        self.batch_size_valid = 1
        self.use_sampler = 0


def denormalize(tensor):
    """
    将 ImageNet 归一化后的 Tensor 反转回 numpy uint8 图片 (H, W, C) BGR格式
    Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C

    # 逆归一化
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # RGB -> BGR (因为 cv2 使用 BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def draw_text(img, text):
    """在图片左上角绘制文字"""
    cv2.putText(img, text, (10, 30), FONT, TEXT_SCALE, TEXT_COLOR, THICKNESS)


def check_split(dataset, mode, output_dir):
    print(f"\nChecking {mode} set...")
    save_dir = os.path.join(output_dir, mode)
    os.makedirs(save_dir, exist_ok=True)

    # 原始 ID 列表，用于从 internal index 映射回真实 Subject ID
    # 直接复制自 dataset_disfa.py 的逻辑
    id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    # 获取该 split 下包含的受试者内部索引列表
    if mode == 'train':
        target_videos = dataset.train_video
    else:
        target_videos = dataset.valid_video

    checked_count = 0

    for subj_idx in target_videos:
        real_subject_id = id_list[subj_idx]

        # 获取该受试者所有的帧索引 (Global indices)
        subj_all_frames = dataset.subject_frames[subj_idx]

        # 【关键】找出该受试者在当前 dataset (可能经过下采样) 中实际存在的帧
        # dataset.original_indices 存储了所有有效的 global_idx
        valid_indices_set = set(dataset.original_indices)
        available_frames = [f for f in subj_all_frames if f in valid_indices_set]

        if not available_frames:
            print(f"[Warning] Subject {real_subject_id} has no frames in {mode} set (possibly filtered out).")
            continue

        # 随机选一帧进行测试
        # 为了能调用 __getitem__，我们需要找到这个 global_idx 在 dataset 中的 list index
        # 这步在海量数据下可能慢，但用于测试脚本没问题
        global_idx_sample = available_frames[len(available_frames) // 2]  # 取中间一帧

        # 找到 __getitem__ 需要的索引 (index)
        try:
            item_index = dataset.original_indices.index(global_idx_sample)
        except ValueError:
            continue

        # 调用 Dataset 的 __getitem__
        curr_tensor, ref_tensor, label, _ = dataset[item_index]

        # 还原图片
        curr_img = denormalize(curr_tensor)
        ref_img = denormalize(ref_tensor)

        # 绘制文字
        au_sum = np.sum(label)
        draw_text(curr_img, f"Subj: {real_subject_id} | Curr (AU Sum: {au_sum:.1f})")
        draw_text(ref_img, f"Subj: {real_subject_id} | Ref (Neutral)")

        # 拼合图片 (竖直拼接)
        # 添加一个黑色分割线
        h, w, c = curr_img.shape
        separator = np.zeros((5, w, c), dtype=np.uint8)
        combined_img = np.vstack([curr_img, separator, ref_img])

        # 保存
        filename = f"Subject_{real_subject_id}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, combined_img)

        checked_count += 1
        print(f"  Saved validation image for Subject {real_subject_id} -> {save_path}")

    print(f"Done. {checked_count} subjects checked for {mode}.")


def main():
    args = parse_args()
    opts = MockOpts(args)

    print("Initializing Dataset (this may take a few seconds to load huge npy files)...")

    # 检查 Valid 集
    valid_dataset = DISFADataset(opts, mode='valid', split=args.split)
    check_split(valid_dataset, 'valid', args.output_dir)

    # 检查 Train 集 (注意 Train 包含下采样逻辑，可能部分帧被丢弃)
    train_dataset = DISFADataset(opts, mode='train', split=args.split)
    check_split(train_dataset, 'train', args.output_dir)

    print(f"\nAll visualization results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()