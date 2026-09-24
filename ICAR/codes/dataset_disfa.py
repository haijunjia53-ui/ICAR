import os
from PIL import Image
import cv2
import json
import argparse
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DISFADataset(data.Dataset):
    def __init__(self, opts, mode='train', split='CCNN'):
        self.debug = 1 if opts.snapshot == 'debug' else 0
        self.mode = mode
        self.split = split
        self.json_dir = opts.json_dir
        self.json_name = opts.json_name

        self.subject_ref_indices = {}
        self.load_data_json()
        self.set_transform()
        self.get_train_valid(self.split)

    def load_data_json(self):
        data_json_path = os.path.join(self.json_dir, self.json_name)
        with open(data_json_path, 'r') as f:
            dataset_json = json.load(f)

        data_path = os.path.join(self.json_dir, dataset_json['image_path'])
        label_path = os.path.join(self.json_dir, dataset_json['label_path'])
        success_path = os.path.join(self.json_dir, dataset_json['success_path'])

        print(f'Loading data from {data_path}...')
        self.full_data = np.load(data_path, mmap_mode='r')
        self.data = self.full_data

        self.label = np.load(label_path, mmap_mode='r')
        self.success = np.load(success_path, mmap_mode='r')

        self.mean = dataset_json['mean']
        self.std = dataset_json['std']
        self.au_number = self.label.shape[-1]

        label_info = dataset_json['label_info']
        subjects = len(label_info['subjects'])
        frames = label_info['frames']

        subjects_start_end = [0]
        for i in range(subjects):
            frame_start = subjects_start_end[i]
            frame_num = frames[i]
            subjects_start_end.append(frame_start + frame_num)

        self.subject_frames = []
        print('Disfa Dataset: Calculating reference frames (Min Intensity Strategy)...')

        for i in range(subjects):
            frame_start = subjects_start_end[i]
            frame_end = subjects_start_end[i + 1]
            frame_index = list(range(frame_start, frame_end))
            self.subject_frames.append(frame_index)

            sub_labels = self.label[frame_index]
            intensities = np.sum(sub_labels, axis=1)
            min_local_idx = np.argmin(intensities)
            min_global_idx = frame_index[min_local_idx]
            self.subject_ref_indices[i] = min_global_idx

        print('Disfa Dataset step1: load data & calc refs ok ...')

    def set_transform(self):
        # 训练和验证都只定义 Normalize，具体的增强逻辑移到 __getitem__ 里做
        if self.debug:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        # 这里的 transforms 只负责最后的 Tensor 化和归一化
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        print('Disfa Dataset step2: set transform ok ...')

    def get_train_valid(self, split):
        id2index = dict()
        id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        for i, id in enumerate(id_list):
            id2index['{}'.format(id)] = i

        if split == 'fold1':
            train_list = [2, 10, 1, 26, 27, 32, 30, 9, 16, 13, 18, 11, 28, 12, 6, 31, 21, 24]
            valid_list = [3, 29, 23, 25, 8, 5, 7, 17, 4]
        elif split == 'fold2':
            train_list = [2, 10, 1, 26, 27, 32, 30, 9, 16, 3, 29, 23, 25, 8, 5, 7, 17, 4]
            valid_list = [13, 18, 11, 28, 12, 6, 31, 21, 24]
        elif split == 'fold3':
            train_list = [13, 18, 11, 28, 12, 6, 31, 21, 24, 3, 29, 23, 25, 8, 5, 7, 17, 4]
            valid_list = [2, 10, 1, 26, 27, 32, 30, 9, 16]
        elif split == 'CCNN':
            train_list = [1, 5, 8, 9, 10, 11, 17, 18, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            valid_list = [2, 3, 4, 6, 7, 12, 13, 16, 23]
        else:
            print('no split method ...')

        self.train_video = [int(id2index['{}'.format(id)]) for id in train_list]
        self.valid_video = [int(id2index['{}'.format(id)]) for id in valid_list]

        train_frames = []
        for video_index in self.train_video:
            train_frames += self.subject_frames[video_index]

        valid_frames = []
        for video_index in self.valid_video:
            valid_frames += self.subject_frames[video_index]

        if self.mode == 'train':
            print(f'[Info] Original Training Frames: {len(train_frames)}')

            # 1. 简单分离
            active_indices = []
            neutral_indices = []

            for idx in train_frames:
                intensity_sum = np.sum(self.label[idx])
                if intensity_sum > 0:
                    active_indices.append(idx)
                else:
                    neutral_indices.append(idx)

            print(f'[Info] Active Frames: {len(active_indices)}, Neutral Frames: {len(neutral_indices)}')

            # 2. 2倍下采样 (这是你 0.61 版本的特征)
            keep_neutral_num = int(len(active_indices) * 2)

            if len(neutral_indices) > keep_neutral_num:
                np.random.seed(42)
                keep_neutral_indices = list(np.random.choice(neutral_indices, keep_neutral_num, replace=False))
            else:
                keep_neutral_indices = neutral_indices

            print(f'[Info] Downsampling Neutrals: Keeping {len(keep_neutral_indices)} neutral frames.')

            final_train_indices = active_indices + keep_neutral_indices
            np.random.seed(42)
            np.random.shuffle(final_train_indices)

            self.original_indices = final_train_indices
            self.img_num = len(final_train_indices)

        else:
            self.original_indices = valid_frames
            self.img_num = len(valid_frames)

        print(f'Disfa Dataset step3: set train valid ok. Mode: {self.mode}, Final Samples: {self.img_num}')

    def __getitem__(self, index):
        global_idx = self.original_indices[index]

        # 1. 读取当前帧
        img_np = self.full_data[global_idx]
        img_pil = Image.fromarray(img_np)

        # 2. 读取参考帧
        subject_id = -1
        for i, frames in enumerate(self.subject_frames):
            if frames[0] <= global_idx <= frames[-1]:
                subject_id = i
                break

        if subject_id != -1:
            ref_idx = self.subject_ref_indices[subject_id]
            ref_np = self.full_data[ref_idx]
            ref_pil = Image.fromarray(ref_np)
        else:
            ref_pil = img_pil.copy()  # 兜底

        # =========================================================
        # 3. 同步数据增强 (Joint Augmentation)
        # =========================================================

        if self.mode == 'train':
            # --- A. 随机水平翻转 ---
            if random.random() > 0.5:
                img_pil = TF.hflip(img_pil)
                ref_pil = TF.hflip(ref_pil)
                # 注意：有些非对称 AU (如只闭左眼) 翻转后语义会变
                # 但对于 DISFA 这种强度估计，通常是可以接受的
                # 如果要严谨，这里还需要同时翻转 label (左右 AU 互换)，但比较复杂，暂时先不做

            # --- B. 随机旋转 (关键！参数必须一致) ---
            angle = transforms.RandomRotation.get_params(degrees=(-15, 15))
            img_pil = TF.rotate(img_pil, angle)
            ref_pil = TF.rotate(ref_pil, angle)

            # --- C. 随机裁剪与缩放 (关键！参数必须一致) ---
            # 随机生成裁剪参数
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img_pil, scale=(0.85, 1.0), ratio=(1.0, 1.0)  # ratio保持1:1防止变形太严重
            )
            # 应用相同的裁剪到两张图
            img_pil = TF.resized_crop(img_pil, i, j, h, w, size=(224, 224))
            ref_pil = TF.resized_crop(ref_pil, i, j, h, w, size=(224, 224))

            # --- D. 颜色抖动 (可以不同步，模拟光照差异) ---
            # 让模型学会：即使参考帧很亮，当前帧很暗，也能减出正确的表情
            color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            img_pil = color_jitter(img_pil)
            ref_pil = color_jitter(ref_pil)

        else:
            # 验证集：只做 Resize (如果原图不是224)
            img_pil = TF.resize(img_pil, (224, 224))
            ref_pil = TF.resize(ref_pil, (224, 224))

        # 4. 转 Tensor 和 归一化
        img_tensor = TF.to_tensor(img_pil)
        img_tensor = self.normalize(img_tensor)

        ref_img_tensor = TF.to_tensor(ref_pil)
        ref_img_tensor = self.normalize(ref_img_tensor)

        label = self.label[global_idx]
        success = self.success[global_idx]

        return img_tensor, ref_img_tensor, label, success

    def __len__(self):
        return self.img_num

class AUBalanceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, use_sampler):
        print('initial balance sampler {}...'.format(use_sampler))
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        au_label_count = {}
        for au in range(dataset.au_number):
            au_label_count[au] = dict()
            for au_value in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                au_label_count[au][au_value] = 0

        for idx in self.indices:
            label = dataset.label[idx]
            for au in range(dataset.au_number):
                val = float(label[au])
                if val in au_label_count[au]:
                    au_label_count[au][val] += 1
                else:
                    au_label_count[au][0.0] += 1

        self.weights = torch.zeros(dataset.au_number, self.num_samples)
        for au in range(dataset.au_number):
            count_dict = au_label_count[au]
            for sample_i, sample_idx in enumerate(self.indices):
                label_sample = dataset.label[sample_idx]
                label_val = float(label_sample[au])
                count = count_dict.get(label_val, 0)
                self.weights[au, sample_i] = 1.0 / (count + 1e-6)

        if use_sampler == 1:
            self.au_weight = torch.FloatTensor([1. / dataset.au_number] * dataset.au_number)
        elif use_sampler == 2:
            self.au_weight = torch.FloatTensor([self.num_samples / (v[1.0] + 1e-6) for k, v in au_label_count.items()])

    def __iter__(self):
        k = torch.multinomial(self.au_weight, 1).item()
        return (self.indices[i] for i in torch.multinomial(self.weights[k], self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def get_data_loader(opts, mode='train', split='CCNN'):
    dataset = DISFADataset(opts, mode, split)

    if mode == 'train':
        batch_size = opts.batch_size_train
        sampler = None
        shuffle = True
    else:
        batch_size = opts.batch_size_valid
        sampler = None
        shuffle = False

    if mode == 'train' and opts.use_sampler:
        sampler = AUBalanceSampler(dataset, opts.use_sampler)
        shuffle = False

    if opts.snapshot == 'debug':
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=0, pin_memory=True
        )

    return data_loader, len(dataset)