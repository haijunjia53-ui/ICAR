#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset_bp4d.py

把原来的 DISFA Dataset 改成 BP4D Dataset。

适配 gene_dataset_bp4d_auint_final.py 生成的数据：
    BP4D_224_images.npy
    BP4D_224_label.npy
    BP4D_224_success.npy
    BP4D_224.json
    BP4D_224_samples.csv

核心变化：
    1. DISFA 是 27 个固定 subject id 的 split；BP4D 改成按 subject 自动划分。
    2. DISFA label_info['frames'] 是按 subject；BP4D 生成脚本里是按 sequence，
       所以这里会根据 label_info['sequences'] 把同一个 subject 的多个 task 合并。
    3. BP4D 默认 AU 是 AU06, AU10, AU12, AU14, AU17，共 5 维。
    4. 保留原来的参考帧策略：每个 subject 选 intensity sum 最小的帧作为 reference。
    5. 保留原来的联合增强、flow、neutral 下采样逻辑。
"""

import os
import json
import math
import random
from PIL import Image

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_csv_list(value):
    if value is None:
        return None
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return items or None


class BP4DDataset(data.Dataset):
    def __init__(self, opts, mode="train", split="fold1"):
        self.debug = 1 if getattr(opts, "snapshot", "") == "debug" else 0
        self.mode = mode
        self.split = split

        self.json_dir = opts.json_dir
        self.json_name = opts.json_name
        self.align_mode = getattr(opts, "align_mode", "none")  # none / flow / implicit

        # BP4D split 相关
        self.valid_subjects = parse_csv_list(getattr(opts, "valid_subjects", None))
        self.valid_ratio = float(getattr(opts, "valid_ratio", 0.2))
        self.fold_num = int(getattr(opts, "fold_num", 5))
        self.split_seed = int(getattr(opts, "split_seed", 42))

        # 训练采样相关
        self.neutral_ratio = float(getattr(opts, "neutral_ratio", 2.0))
        self.filter_success = int(getattr(opts, "filter_success", 0))

        self.subject_ref_indices = {}
        self.subject_frames = []
        self.subject_names = []
        self.index_to_subject_id = None

        # =========================
        # Temporal configuration
        # =========================
        self.temporal_window = int(
            getattr(opts, "temporal_window", 9)
        )
        self.temporal_stride_train = int(
            getattr(opts, "temporal_stride_train", 3)
        )
        self.temporal_stride_valid = int(
            getattr(opts, "temporal_stride_valid", 1)
        )

        if self.temporal_window <= 0:
            raise ValueError("temporal_window 必须大于 0")

        if self.temporal_window % 2 == 0:
            raise ValueError(
                f"temporal_window 必须是奇数，当前为 {self.temporal_window}"
            )

        if self.temporal_stride_train <= 0:
            raise ValueError("temporal_stride_train 必须大于 0")

        if self.temporal_stride_valid <= 0:
            raise ValueError("temporal_stride_valid 必须大于 0")

        # 每个独立 task/sequence 的信息
        self.sequence_frames = []
        self.sequence_names = []
        self.sequence_subject_ids = []

        self.index_to_sequence_id = None
        self.index_to_frame_pos = None

        # Dataset 的样本单位改成 temporal window
        self.windows = []

        self.load_data_json()
        self.set_transform()
        self.get_train_valid(self.split)

    def load_data_json(self):
        data_json_path = os.path.join(self.json_dir, self.json_name)
        with open(data_json_path, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)

        data_path = os.path.join(self.json_dir, dataset_json["image_path"])
        label_path = os.path.join(self.json_dir, dataset_json["label_path"])
        success_path = os.path.join(self.json_dir, dataset_json["success_path"])

        print(f"Loading BP4D data from {data_path}...")
        self.full_data = np.load(data_path, mmap_mode="r")
        self.data = self.full_data

        self.label = np.load(label_path, mmap_mode="r")
        self.success = np.load(success_path, mmap_mode="r")

        self.dataset_mean = dataset_json.get("mean", [0.485, 0.456, 0.406])
        self.dataset_std = dataset_json.get("std", [0.229, 0.224, 0.225])

        self.au_ids = dataset_json.get(
            "aus",
            ["AU06", "AU10", "AU12", "AU14", "AU17"]
        )
        self.au_number = int(self.label.shape[-1])

        # 让 train_bp4d.py 可以把 au_number 写回 opts，给 model_bp4d.py 使用
        print(f"BP4D AU order: {self.au_ids}")
        print(f"BP4D label dim: {self.au_number}")

        label_info = dataset_json["label_info"]

        # gene_dataset_bp4d_auint_final.py 生成的是按 sequence 存储：
        # label_info["sequences"] = [
        #   {"name": "F001_T1", "subject": "F001", "task": "T1", "frames": xxx},
        #   ...
        # ]
        if "sequences" not in label_info:
            raise RuntimeError(
                "时序模型要求 json 的 label_info 中包含 sequences。"
                "当前 json 只有 subject-level frames，无法保证窗口不跨 task。"
            )

        self._build_subject_frames_from_bp4d_sequences(
            label_info
        )

        assert len(self.index_to_subject_id) == self.label.shape[0], (
            f"index_to_subject_id length {len(self.index_to_subject_id)} "
            f"!= label rows {self.label.shape[0]}"
        )

        # 每个 subject 选一张 neutral/reference 帧：
        # 原策略：标签强度和最小的帧
        print("BP4D Dataset: Calculating reference frames (Min Intensity Strategy)...")
        for subject_id, frame_index in enumerate(self.subject_frames):
            if len(frame_index) == 0:
                continue
            sub_labels = self.label[frame_index]
            intensities = np.sum(sub_labels, axis=1)
            min_local_idx = int(np.argmin(intensities))
            min_global_idx = int(frame_index[min_local_idx])
            self.subject_ref_indices[subject_id] = min_global_idx

        print("BP4D Dataset step1: load data & calc refs ok ...")

    def _build_subject_frames_from_bp4d_sequences(self, label_info):
        """
        同时构造：
            subject_frames:
                用于 subject split 和 reference frame。

            sequence_frames:
                用于时序窗口，绝不跨 task。

            index_to_subject_id:
                global frame index -> subject id。

            index_to_sequence_id:
                global frame index -> sequence id。

            index_to_frame_pos:
                global frame index -> 该 sequence 内的位置。
        """
        sequences = label_info["sequences"]

        # 先获取所有有效 subject
        subject_names_in_sequences = []

        for seq in sequences:
            frame_num = int(seq.get("frames", 0))
            if frame_num <= 0:
                continue

            sequence_name = str(seq.get("name", ""))
            subject_name = str(
                seq.get(
                    "subject",
                    sequence_name.split("_")[0],
                )
            ).upper()

            if subject_name not in subject_names_in_sequences:
                subject_names_in_sequences.append(subject_name)

        # 优先沿用 json 中的 subject 顺序
        json_subjects = [
            str(s).upper()
            for s in label_info.get("subjects", [])
        ]

        ordered_subjects = [
            s
            for s in json_subjects
            if s in subject_names_in_sequences
        ]

        for subject_name in subject_names_in_sequences:
            if subject_name not in ordered_subjects:
                ordered_subjects.append(subject_name)

        self.subject_names = ordered_subjects

        subject_to_id = {
            subject_name: subject_id
            for subject_id, subject_name
            in enumerate(self.subject_names)
        }

        subject_to_frames = {
            subject_name: []
            for subject_name in self.subject_names
        }

        self.sequence_frames = []
        self.sequence_names = []
        self.sequence_subject_ids = []

        frame_count = int(self.label.shape[0])

        self.index_to_subject_id = np.full(
            (frame_count,),
            -1,
            dtype=np.int64,
        )
        self.index_to_sequence_id = np.full(
            (frame_count,),
            -1,
            dtype=np.int64,
        )
        self.index_to_frame_pos = np.full(
            (frame_count,),
            -1,
            dtype=np.int64,
        )

        global_start = 0

        for seq in sequences:
            frame_num = int(seq.get("frames", 0))
            if frame_num <= 0:
                continue

            sequence_name = str(
                seq.get(
                    "name",
                    f"sequence_{len(self.sequence_frames)}",
                )
            )

            subject_name = str(
                seq.get(
                    "subject",
                    sequence_name.split("_")[0],
                )
            ).upper()

            if subject_name not in subject_to_id:
                raise RuntimeError(
                    f"sequence={sequence_name} 的 subject={subject_name} "
                    f"不在 subject_to_id 中"
                )

            subject_id = subject_to_id[subject_name]
            sequence_id = len(self.sequence_frames)

            global_end = global_start + frame_num

            if global_end > frame_count:
                raise RuntimeError(
                    f"sequence={sequence_name} 超出 label 长度："
                    f"{global_end} > {frame_count}"
                )

            frame_indices = list(
                range(global_start, global_end)
            )

            self.sequence_names.append(sequence_name)
            self.sequence_frames.append(frame_indices)
            self.sequence_subject_ids.append(subject_id)

            subject_to_frames[subject_name].extend(
                frame_indices
            )

            self.index_to_subject_id[
            global_start:global_end
            ] = subject_id

            self.index_to_sequence_id[
            global_start:global_end
            ] = sequence_id

            self.index_to_frame_pos[
            global_start:global_end
            ] = np.arange(
                frame_num,
                dtype=np.int64,
            )

            global_start = global_end

        if global_start != frame_count:
            raise RuntimeError(
                f"label_info sequence 累计帧数 {global_start} "
                f"!= label rows {frame_count}"
            )

        self.subject_frames = [
            sorted(subject_to_frames[subject_name])
            for subject_name in self.subject_names
        ]

        if np.any(self.index_to_subject_id < 0):
            raise RuntimeError(
                "存在没有分配 subject_id 的帧"
            )

        if np.any(self.index_to_sequence_id < 0):
            raise RuntimeError(
                "存在没有分配 sequence_id 的帧"
            )

        print(
            f"[Temporal Meta] subjects={len(self.subject_names)}, "
            f"sequences={len(self.sequence_frames)}, "
            f"frames={frame_count}"
        )

    def _build_subject_frames_from_subject_frames(self, label_info):
        self.subject_names = [str(s).upper() for s in label_info["subjects"]]
        frames = [int(x) for x in label_info["frames"]]

        self.subject_frames = []
        self.index_to_subject_id = np.full((self.label.shape[0],), -1, dtype=np.int64)

        start = 0
        for sid, frame_num in enumerate(frames):
            indices = list(range(start, start + frame_num))
            self.subject_frames.append(indices)
            self.index_to_subject_id[indices] = sid
            start += frame_num

        if start != self.label.shape[0]:
            raise RuntimeError(
                f"label_info frames 累计帧数 {start} != label rows {self.label.shape[0]}"
            )

    def set_transform(self):
        # 默认使用 ImageNet 归一化，和 ResNet 预训练权重更匹配。
        # 如果想使用 gene_dataset 统计出的 mean/std，运行时加 --norm_mode dataset
        norm_mode = getattr(self, "norm_mode", None)
        if norm_mode is None:
            norm_mode = "imagenet"

        if self.debug:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif norm_mode == "dataset":
            self.mean = self.dataset_mean
            self.std = self.dataset_std
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        print("BP4D Dataset step2: set transform ok ...")

    def _subject_split_by_valid_subjects(self):
        valid_names = {s.upper() for s in self.valid_subjects}
        train_ids = []
        valid_ids = []

        for i, name in enumerate(self.subject_names):
            if name in valid_names:
                valid_ids.append(i)
            else:
                train_ids.append(i)

        if len(valid_ids) == 0:
            raise RuntimeError(
                f"--valid_subjects={self.valid_subjects} 没有匹配到任何 subject。"
                f"可用 subject 示例：{self.subject_names[:10]}"
            )

        return train_ids, valid_ids

    def _subject_split_by_fold(self, split):
        """
        按 subject 做 deterministic K-fold。
        split 可以是 fold1/fold2/...，也可以是 CCNN。
        """
        subject_ids = list(range(len(self.subject_names)))

        rng = random.Random(self.split_seed)
        rng.shuffle(subject_ids)

        split_lower = str(split).lower()

        if split_lower == "all":
            return subject_ids, subject_ids

        if split_lower.startswith("fold"):
            digits = "".join(ch for ch in split_lower if ch.isdigit())
            fold_idx = int(digits) - 1 if digits else 0
        else:
            # 兼容原来默认的 CCNN，但这里不再使用 DISFA 固定名单；
            # CCNN 等价于第 1 折。
            fold_idx = 0

        fold_idx = fold_idx % max(1, self.fold_num)
        fold_size = int(math.ceil(len(subject_ids) / float(self.fold_num)))

        valid_start = fold_idx * fold_size
        valid_end = min(len(subject_ids), valid_start + fold_size)

        valid_ids = subject_ids[valid_start:valid_end]
        train_ids = [sid for sid in subject_ids if sid not in valid_ids]

        # 如果 subject 很少，避免 train 为空
        if len(train_ids) == 0:
            train_ids = valid_ids

        return train_ids, valid_ids

    def _build_temporal_windows(
            self,
            allowed_subject_ids,
    ):
        """
        以中心帧为监督目标构造固定长度窗口。

        训练：
            stride = temporal_stride_train

        验证：
            stride = temporal_stride_valid，推荐固定为 1
        """
        allowed_subject_ids = {
            int(subject_id)
            for subject_id in allowed_subject_ids
        }

        if self.mode == "train":
            stride = self.temporal_stride_train
        else:
            stride = self.temporal_stride_valid

        half_window = self.temporal_window // 2
        windows = []

        for sequence_id, sequence_indices in enumerate(
                self.sequence_frames
        ):
            if len(sequence_indices) == 0:
                continue

            subject_id = int(
                self.sequence_subject_ids[sequence_id]
            )

            if subject_id not in allowed_subject_ids:
                continue

            frame_num = len(sequence_indices)

            for center_pos in range(
                    0,
                    frame_num,
                    stride,
            ):
                center_global_idx = int(
                    sequence_indices[center_pos]
                )

                # filter_success 只决定中心帧是否作为监督样本。
                # 不能删除窗口内部帧，否则会破坏连续时序。
                if self.filter_success:
                    if int(self.success[center_global_idx]) != 1:
                        continue

                window_indices = []

                for offset in range(
                        -half_window,
                        half_window + 1,
                ):
                    temporal_pos = center_pos + offset

                    # 边界复制，确保每个样本固定 T 帧
                    temporal_pos = max(
                        0,
                        min(frame_num - 1, temporal_pos),
                    )

                    global_idx = int(
                        sequence_indices[temporal_pos]
                    )
                    window_indices.append(global_idx)

                if len(window_indices) != self.temporal_window:
                    raise RuntimeError(
                        f"窗口长度错误："
                        f"{len(window_indices)} "
                        f"!= {self.temporal_window}"
                    )

                windows.append(
                    {
                        "sequence_id": int(sequence_id),
                        "sequence_name": self.sequence_names[
                            sequence_id
                        ],
                        "subject_id": subject_id,
                        "center_pos": int(center_pos),
                        "center_global_idx": center_global_idx,
                        "indices": window_indices,
                    }
                )

        return windows

    def get_train_valid(self, split):
        if self.valid_subjects is not None:
            (
                self.train_video,
                self.valid_video,
            ) = self._subject_split_by_valid_subjects()
        else:
            (
                self.train_video,
                self.valid_video,
            ) = self._subject_split_by_fold(split)

        train_names = [
            self.subject_names[i]
            for i in self.train_video
        ]
        valid_names = [
            self.subject_names[i]
            for i in self.valid_video
        ]

        print(
            f"[BP4D Split] train subjects "
            f"({len(train_names)}): {train_names}"
        )
        print(
            f"[BP4D Split] valid subjects "
            f"({len(valid_names)}): {valid_names}"
        )

        if self.mode == "train":
            allowed_subject_ids = self.train_video
        else:
            allowed_subject_ids = self.valid_video

        self.windows = self._build_temporal_windows(
            allowed_subject_ids
        )

        print(
            f"[Temporal] Original windows: "
            f"{len(self.windows)}"
        )

        # 保留你当前 neutral_ratio 的含义，
        # 但下采样单位从 frame 改成 window。
        # 窗口内部帧绝不删除。
        if self.mode == "train":
            active_windows = []
            neutral_windows = []

            for window in self.windows:
                center_idx = window["center_global_idx"]
                center_label = self.label[center_idx]

                intensity_sum = float(
                    np.sum(center_label)
                )

                if intensity_sum > 0:
                    active_windows.append(window)
                else:
                    neutral_windows.append(window)

            print(
                f"[Temporal] Active center windows: "
                f"{len(active_windows)}"
            )
            print(
                f"[Temporal] Neutral center windows: "
                f"{len(neutral_windows)}"
            )

            if self.neutral_ratio > 0:
                keep_neutral_num = int(
                    len(active_windows)
                    * self.neutral_ratio
                )

                if len(neutral_windows) > keep_neutral_num:
                    rng = np.random.default_rng(
                        self.split_seed
                    )

                    selected_indices = rng.choice(
                        len(neutral_windows),
                        size=keep_neutral_num,
                        replace=False,
                    )

                    neutral_windows = [
                        neutral_windows[int(i)]
                        for i in selected_indices
                    ]

                print(
                    f"[Temporal] Keep neutral windows: "
                    f"{len(neutral_windows)}, "
                    f"neutral_ratio={self.neutral_ratio}"
                )

                self.windows = (
                        active_windows
                        + neutral_windows
                )

            # 这里只打乱窗口顺序，
            # 不会破坏每个窗口内部的连续帧。
            rng = random.Random(self.split_seed)
            rng.shuffle(self.windows)

        self.img_num = len(self.windows)

        print(
            f"BP4D temporal dataset ready: "
            f"mode={self.mode}, "
            f"T={self.temporal_window}, "
            f"samples={self.img_num}"
        )

    def _sample_window_augmentation(
            self,
            first_image,
    ):
        """
        为整个窗口采样一次几何增强参数。
        """
        if self.mode != "train":
            return {
                "do_flip": False,
                "angle": 0.0,
                "crop_params": None,
            }

        do_flip = random.random() > 0.5

        angle = transforms.RandomRotation.get_params(
            degrees=(-15, 15)
        )

        crop_params = (
            transforms.RandomResizedCrop.get_params(
                first_image,
                scale=(0.85, 1.0),
                ratio=(1.0, 1.0),
            )
        )

        return {
            "do_flip": do_flip,
            "angle": float(angle),
            "crop_params": crop_params,
        }

    def _load_frame_pair(
            self,
            global_idx,
            aug_params,
    ):
        """
        加载一个 current/reference pair。

        同一窗口内的每次调用使用完全相同 aug_params。
        """
        global_idx = int(global_idx)

        # current
        img_np = self.full_data[global_idx]
        img_pil = Image.fromarray(img_np)

        # reference
        subject_id = int(
            self.index_to_subject_id[global_idx]
        )

        if (
                subject_id >= 0
                and subject_id in self.subject_ref_indices
        ):
            ref_idx = int(
                self.subject_ref_indices[subject_id]
            )
            ref_np = self.full_data[ref_idx]
            ref_pil = Image.fromarray(ref_np)
        else:
            ref_pil = img_pil.copy()

        if self.mode == "train":
            do_flip = bool(
                aug_params["do_flip"]
            )
            angle = float(
                aug_params["angle"]
            )
            crop_params = aug_params[
                "crop_params"
            ]

            if do_flip:
                img_pil = TF.hflip(img_pil)
                ref_pil = TF.hflip(ref_pil)

            img_pil = TF.rotate(
                img_pil,
                angle,
            )
            ref_pil = TF.rotate(
                ref_pil,
                angle,
            )

            i, j, h, w = crop_params

            img_pil = TF.resized_crop(
                img_pil,
                i,
                j,
                h,
                w,
                size=(224, 224),
            )

            ref_pil = TF.resized_crop(
                ref_pil,
                i,
                j,
                h,
                w,
                size=(224, 224),
            )

            # 第一版时序实验先关闭 ColorJitter。
            # 否则需要自己采样固定参数，不能逐帧随机调用。
        else:
            img_pil = TF.resize(
                img_pil,
                (224, 224),
            )
            ref_pil = TF.resize(
                ref_pil,
                (224, 224),
            )

        flow_tensor = torch.zeros(
            2,
            224,
            224,
            dtype=torch.float32,
        )

        if self.align_mode == "flow":
            img_gray = np.array(
                img_pil.convert("L")
            )
            ref_gray = np.array(
                ref_pil.convert("L")
            )

            flow = cv2.calcOpticalFlowFarneback(
                ref_gray,
                img_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            flow_tensor = torch.from_numpy(
                flow
            ).permute(2, 0, 1).float()

        img_tensor = TF.to_tensor(img_pil)
        img_tensor = self.normalize(img_tensor)

        ref_tensor = TF.to_tensor(ref_pil)
        ref_tensor = self.normalize(ref_tensor)

        return (
            img_tensor,
            ref_tensor,
            flow_tensor,
        )

    def __getitem__(self, index):
        window = self.windows[index]

        window_indices = window["indices"]
        center_global_idx = int(
            window["center_global_idx"]
        )
        subject_id = int(
            window["subject_id"]
        )
        sequence_id = int(
            window["sequence_id"]
        )
        center_pos = int(
            window["center_pos"]
        )

        first_image = Image.fromarray(
            self.full_data[window_indices[0]]
        )

        aug_params = self._sample_window_augmentation(
            first_image
        )

        image_sequence = []
        reference_sequence = []
        flow_sequence = []

        for global_idx in window_indices:
            (
                img_tensor,
                ref_tensor,
                flow_tensor,
            ) = self._load_frame_pair(
                global_idx,
                aug_params,
            )

            image_sequence.append(img_tensor)
            reference_sequence.append(ref_tensor)
            flow_sequence.append(flow_tensor)

        image_sequence = torch.stack(
            image_sequence,
            dim=0,
        )

        reference_sequence = torch.stack(
            reference_sequence,
            dim=0,
        )

        flow_sequence = torch.stack(
            flow_sequence,
            dim=0,
        )

        center_label = np.asarray(
            self.label[center_global_idx],
            dtype=np.float32,
        )

        center_success = np.asarray(
            self.success[center_global_idx]
        ).astype(np.int64)

        return (
            image_sequence,  # [T,3,224,224]
            reference_sequence,  # [T,3,224,224]
            flow_sequence,  # [T,2,224,224]
            center_label,  # [K]
            center_success,
            subject_id,
            sequence_id,
            center_pos,
            center_global_idx,
        )

    def __len__(self):
        return self.img_num


class AUBalanceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, use_sampler):
        print(f"initial BP4D balance sampler {use_sampler}...")

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
        self.au_number = dataset.au_number

        au_label_count = {
            au: {v: 0 for v in [0, 1, 2, 3, 4, 5]}
            for au in range(dataset.au_number)
        }

        for local_idx in self.indices:
            label = self._get_label(dataset, local_idx)
            for au in range(dataset.au_number):
                val = int(round(float(label[au])))
                val = max(0, min(5, val))
                au_label_count[au][val] += 1

        self.weights = torch.zeros(dataset.au_number, self.num_samples)

        for au in range(dataset.au_number):
            for sample in self.indices:
                label_sample = self._get_label(dataset, sample)
                label_sample_au = int(round(float(label_sample[au])))
                label_sample_au = max(0, min(5, label_sample_au))
                count = max(1, au_label_count[au][label_sample_au])
                self.weights[au, sample] = 1.0 / float(count)

        if use_sampler == 1:
            self.au_weight = torch.FloatTensor([1.0 / dataset.au_number] * dataset.au_number)
        elif use_sampler == 2:
            # 优先关注 active 样本少的 AU
            active_counts = []
            for au in range(dataset.au_number):
                active_count = sum(au_label_count[au][v] for v in [1, 2, 3, 4, 5])
                active_counts.append(max(1, active_count))
            raw = torch.FloatTensor([self.num_samples / c for c in active_counts])
            self.au_weight = raw / raw.sum()
        else:
            self.au_weight = torch.FloatTensor([1.0 / dataset.au_number] * dataset.au_number)

    def _get_label(self, dataset, local_idx):
        global_idx = dataset.original_indices[local_idx]
        return dataset.label[global_idx]

    def __iter__(self):
        k = torch.multinomial(self.au_weight, 1).item()
        sampled = torch.multinomial(self.weights[k], self.num_samples, replacement=True)
        return (self.indices[i] for i in sampled)

    def __len__(self):
        return self.num_samples


def get_data_loader(
    opts,
    mode="train",
    split="fold1",
):
    dataset = BP4DDataset(
        opts,
        mode,
        split,
    )

    if mode == "train":
        batch_size = opts.batch_size_train
        shuffle = True
    else:
        batch_size = opts.batch_size_valid
        shuffle = False

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(
            getattr(opts, "num_workers", 4)
        ),
        pin_memory=True,
        drop_last=(mode == "train"),
        persistent_workers=(
            int(getattr(opts, "num_workers", 4)) > 0
        ),
    )

    return data_loader, len(dataset)
