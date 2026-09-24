import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def flow_warp(x, flow):
    """
    x:    (N,C,H,W) - Feature map to be warped
    flow: (N,2,H,W) - Displacement (dx, dy) in feature map pixel coordinates
    """
    N, C, H, W = x.size()
    # base grid: (H,W,2) in pixel coords
    ys, xs = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack((xs, ys), dim=-1).float()  # (H,W,2)
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N,H,W,2)

    # Apply flow to grid
    vgrid = grid + flow.permute(0, 2, 3, 1)  # (N,H,W,2)

    # Normalize to [-1,1] for grid_sample
    vgrid_x = (vgrid[..., 0] + 0.5) / W * 2.0 - 1.0
    vgrid_y = (vgrid[..., 1] + 0.5) / H * 2.0 - 1.0
    vgrid_norm = torch.stack((vgrid_x, vgrid_y), dim=-1)

    return F.grid_sample(x, vgrid_norm, mode='bilinear', padding_mode='border', align_corners=False)


class LearnableFeatureWarping(nn.Module):
    """
    新的隐式对齐模块：基于卷积预测特征层的光流 (dx, dy)
    替代了原来的 Cross-Attention，能更好地保留空间结构信息。
    """

    def __init__(self, feature_dim=1024):
        super().__init__()
        # 输入是 concat(current, ref)，所以通道数 * 2
        self.offset_estimator = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 2, kernel_size=3, padding=1)  # Output: (N, 2, H, W) -> dx, dy
        )

        # 初始化：最后一层权重置0，确保初始状态下 output flow 为 0 (即 Identity Mapping)
        # 这样训练开始时不会因为随机初始化导致特征乱飞
        self.offset_estimator[-1].weight.data.zero_()
        self.offset_estimator[-1].bias.data.zero_()

    def forward(self, x_curr, x_ref):
        # x_curr, x_ref: [N, C, H, W]

        # 1. 拼接特征
        combined = torch.cat([x_curr, x_ref], dim=1)  # [N, 2C, H, W]

        # 2. 预测偏移量 (Flow in Feature Space)
        estimated_flow = self.offset_estimator(combined)  # [N, 2, H, W]

        # 3. 执行 Warp
        aligned_ref = flow_warp(x_ref, estimated_flow)

        return aligned_ref, estimated_flow


class PatchGenerator(nn.Module):
    def __init__(self, patch_number, in_channels):
        super(PatchGenerator, self).__init__()
        self.patch_number = patch_number
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 2 * 3 * self.patch_number),
        )
        # 初始化为预定义的 Patch 位置
        path_postion = [1, 0, 0, 0, 1 / 3, -13 / 20,
                        3 / 4, 0, 0, 0, 1 / 4, -1 / 10,
                        1, 0, 0, 0, 1 / 3, 5 / 10]
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(path_postion, dtype=torch.float))

    def forward(self, x, return_theta: bool = False):
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (1, 1))
        xs = torch.flatten(xs, 1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, self.patch_number, 2, 3)

        output = []
        for i in range(self.patch_number):
            stripe = theta.narrow(1, i, 1).squeeze(1)
            grid = F.affine_grid(stripe, x.size(), align_corners=False)
            output.append(F.grid_sample(x, grid, align_corners=False))

        if return_theta:
            return output, theta
        return output


class LayerNorm(nn.Module):
    def __init__(self, individual_featured):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(individual_featured))
        self.b = nn.Parameter(torch.zeros(individual_featured))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


def dot_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_number, patch_number, individual_featured):
        super(MultiHeadedAttention, self).__init__()
        assert individual_featured % head_number == 0
        self.d_k = individual_featured // head_number
        self.h = head_number
        linear = []
        for i in range(4):
            linear += [nn.Linear(individual_featured, individual_featured)]
        self.linears = nn.Sequential(*linear)

    def forward(self, query, key, value):
        N = query.size(0)
        query, key, value = \
            [l(x).view(N, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x = dot_attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, individual_featured):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(individual_featured, 2 * individual_featured)
        self.w_2 = nn.Linear(2 * individual_featured, individual_featured)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, head_number, patch_number, individual_featured):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = MultiHeadedAttention(head_number, patch_number, individual_featured)
        self.feedforward = PositionwiseFeedForward(individual_featured)
        self.norm_attention = LayerNorm(individual_featured)
        self.norm_feedforward = LayerNorm(individual_featured)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        residual = x
        x = self.norm_attention(x)
        x = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual
        residual = x
        x = self.norm_feedforward(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + residual
        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SelfAttention(nn.Module):
    def __init__(self, block_num, head_number, patch_number, individual_featured):
        super(SelfAttention, self).__init__()
        block = MultiHeadAttentionBlock(head_number, patch_number, individual_featured)
        self.layers = clones(block, block_num)
        self.norm = LayerNorm(individual_featured)
        self.pos = nn.Embedding(patch_number, individual_featured).to(device)

    def forward(self, x):
        x = torch.stack(x, dim=1)
        N = x.shape[0]
        pos_num = torch.LongTensor([0, 1, 2]).to(device)
        pos_embedding = self.pos(pos_num)
        pos_embedding = pos_embedding.unsqueeze(0).repeat(N, 1, 1)
        x = x + pos_embedding
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = torch.split(x, 1, dim=1)
        x = [x_i.squeeze(1) for x_i in x]
        return x

class TemporalResidualBiGRU(nn.Module):
    """
    输入：
        frame_features:    [B,T,D]
        frame_predictions: [B,T,K]

    输出：
        中心帧修正后的预测 [B,K]
    """

    def __init__(
        self,
        feature_dim,
        au_number,
        hidden_dim=256,
        dropout=0.2,
        delta_scale=0.1,
    ):
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.au_number = int(au_number)
        self.hidden_dim = int(hidden_dim)

        temporal_input_dim = (
            self.feature_dim
            + self.au_number
        )

        self.input_norm = nn.LayerNorm(
            temporal_input_dim
        )

        self.gru = nn.GRU(
            input_size=temporal_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.delta_head = nn.Sequential(
            nn.LayerNorm(
                self.hidden_dim * 2
            ),
            nn.Linear(
                self.hidden_dim * 2,
                self.hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_dim,
                self.au_number,
            ),
        )

        # 初始化时 temporal delta = 0，
        # 模型预测与原单帧模型一致。
        final_layer = self.delta_head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        self.delta_scale = float(delta_scale)

    def forward(
        self,
        frame_features,
        frame_predictions,
    ):
        if frame_features.dim() != 3:
            raise ValueError(
                f"frame_features 应为 [B,T,D]，"
                f"实际为 {frame_features.shape}"
            )

        if frame_predictions.dim() != 3:
            raise ValueError(
                f"frame_predictions 应为 [B,T,K]，"
                f"实际为 {frame_predictions.shape}"
            )

        if frame_features.shape[:2] != (
            frame_predictions.shape[:2]
        ):
            raise ValueError(
                "frame_features 与 frame_predictions "
                "的 B、T 不一致"
            )

        # 第一阶段让 prediction context 不反向影响
        # 原单帧回归支路。
        temporal_input = torch.cat(
            [
                frame_features,
                frame_predictions.detach(),
            ],
            dim=-1,
        )

        temporal_input = self.input_norm(
            temporal_input
        )

        temporal_output, _ = self.gru(
            temporal_input
        )

        center = temporal_output.size(1) // 2

        temporal_center = temporal_output[
            :, center, :
        ]

        delta = self.delta_head(
            temporal_center
        )

        center_frame_prediction = (
            frame_predictions[:, center, :]
        )

        prediction = (
            center_frame_prediction
            + self.delta_scale * delta
        )

        return torch.relu(prediction)


class AU_IMF_Net(nn.Module):
    def __init__(self, opts):
        super(AU_IMF_Net, self).__init__()

        self.encoder = ResNetFeatureExtractor(pretrained=True)
        self.backbone_out_channels = 1024

        self.align_mode = getattr(opts, 'align_mode', 'none')

        # 1. 对齐模块选择
        if self.align_mode == 'implicit':
            # 使用 LearnableFeatureWarping
            self.alignment = LearnableFeatureWarping(feature_dim=self.backbone_out_channels)
        else:
            # flow 或 none 模式不需要这个模块，用 Identity 占位或不初始化
            self.alignment = nn.Identity()

        self.patch_number = opts.patch_number
        self.individual_featured = opts.individual_featured

        self.new = nn.ModuleList()
        down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone_out_channels, self.individual_featured, 1),
                nn.BatchNorm2d(self.individual_featured),
                nn.ReLU(inplace=True)
            ) for _ in range(self.patch_number)
        ])
        self.new.add_module('down', down)

        self.target_task = opts.target_task

        # BP4D 默认是 5 个 intensity AU:
        # [AU06, AU10, AU12, AU14, AU17]
        # 原 DISFA 代码固定 patch_au_num = [5, 1, 6]，总输出 12 维；
        # BP4D 必须改成总和为 5，否则 pred 和 label 维度不匹配。
        self.au_number = int(getattr(opts, "au_number", 5))
        patch_au_num_arg = getattr(opts, "patch_au_num", "1,1,3")
        if isinstance(patch_au_num_arg, str):
            patch_au_num = [int(x.strip()) for x in patch_au_num_arg.split(",") if x.strip()]
        else:
            patch_au_num = list(patch_au_num_arg)

        if len(patch_au_num) != self.patch_number:
            raise ValueError(
                f"patch_au_num 长度必须等于 patch_number。"
                f"当前 patch_au_num={patch_au_num}, patch_number={self.patch_number}"
            )

        if sum(patch_au_num) != self.au_number:
            raise ValueError(
                f"patch_au_num 总和必须等于 au_number。"
                f"当前 patch_au_num={patch_au_num}, sum={sum(patch_au_num)}, au_number={self.au_number}"
            )

        self.patch_au_num = patch_au_num
        print(f"[Model BP4D] au_number={self.au_number}, patch_au_num={self.patch_au_num}")

        if 'regress' in self.target_task:
            fc_regressor = nn.ModuleList([
                nn.Linear(self.individual_featured, patch_au_num[i]) for i in range(self.patch_number)
            ])
            self.new.add_module('regress', fc_regressor)

        if 'uncertain' in self.target_task:
            fc_uncertain = nn.ModuleList([
                nn.Linear(self.individual_featured, patch_au_num[i]) for i in range(self.patch_number)
            ])
            self.new.add_module('uncertain', fc_uncertain)

        if 'classify' in self.target_task:
            fc_classifier = nn.ModuleList([
                nn.Linear(self.individual_featured, 6 * patch_au_num[i]) for i in range(self.patch_number)
            ])
            self.new.add_module('classify', fc_classifier)

        self.patch_proposal = PatchGenerator(self.patch_number, self.backbone_out_channels)
        self.self_attention = SelfAttention(opts.block_num, opts.head_number,
                                            opts.patch_number, opts.individual_featured)
        self.frame_feature_dim = (
                self.patch_number
                * self.individual_featured
        )

        self.temporal_head = TemporalResidualBiGRU(
            feature_dim=self.frame_feature_dim,
            au_number=self.au_number,
            hidden_dim=int(
                getattr(opts, "temporal_hidden", 256)
            ),
            dropout=float(
                getattr(opts, "temporal_dropout", 0.2)
            ),
            delta_scale=float(
                getattr(opts, "temporal_delta_scale", 0.1)
            ),
        )

        print(
            f"[Temporal Model] "
            f"frame_feature_dim={self.frame_feature_dim}, "
            f"hidden={getattr(opts, 'temporal_hidden', 256)}, "
            f"delta_scale={getattr(opts, 'temporal_delta_scale', 0.1)}"
        )

    def extract_frame_outputs(
            self,
            x_current,
            x_reference,
            flow=None,
    ):
        """
        x_current/x_reference:
            [N,3,H,W]

        返回：
            frame_feature: [N, patch_number * individual_featured]
            pred_mean:     [N,K]
            pred_std:      [N,K] 或 None
            pred_logits:   [N,K*6] 或 None
        """
        f_c = self.encoder(x_current)
        f_r = self.encoder(x_reference)

        estimated_flow = None

        if self.align_mode == "implicit":
            (
                aligned_ref,
                estimated_flow,
            ) = self.alignment(
                f_c,
                f_r,
            )

        elif self.align_mode == "flow":
            if flow is None:
                raise ValueError(
                    "align_mode=flow requires flow input"
                )

            _, _, feature_h, feature_w = f_c.shape
            flow_h, flow_w = flow.shape[-2:]

            flow_feat = F.interpolate(
                flow,
                size=(feature_h, feature_w),
                mode="bilinear",
                align_corners=False,
            )

            flow_feat[:, 0, :, :] *= (
                    feature_w / float(flow_w)
            )

            flow_feat[:, 1, :, :] *= (
                    feature_h / float(flow_h)
            )

            aligned_ref = flow_warp(
                f_r,
                flow_feat,
            )
        else:
            aligned_ref = f_r

        motion_feature = f_c - aligned_ref

        patches = self.patch_proposal(
            motion_feature
        )

        patch_features = []

        for patch_id in range(self.patch_number):
            feature = F.adaptive_avg_pool2d(
                patches[patch_id],
                output_size=1,
            )

            feature = self.new.down[
                patch_id
            ](feature)

            feature = torch.flatten(
                feature,
                1,
            )

            patch_features.append(feature)

        patch_features = self.self_attention(
            patch_features
        )

        # [N, 3×256] = [N,768]
        frame_feature = torch.cat(
            patch_features,
            dim=1,
        )

        pred_mean_list = []
        pred_std_list = []
        pred_logits_list = []

        for patch_id in range(self.patch_number):
            patch_feature = patch_features[
                patch_id
            ]

            if "regress" in self.target_task:
                pred_mean_list.append(
                    self.new.regress[
                        patch_id
                    ](patch_feature)
                )

            if "uncertain" in self.target_task:
                pred_std_list.append(
                    self.new.uncertain[
                        patch_id
                    ](patch_feature)
                )

            if "classify" in self.target_task:
                pred_logits_list.append(
                    self.new.classify[
                        patch_id
                    ](patch_feature)
                )

        if "regress" in self.target_task:
            pred_mean = torch.cat(
                pred_mean_list,
                dim=1,
            )
            pred_mean = torch.relu(pred_mean)
        else:
            pred_mean = None

        if "uncertain" in self.target_task:
            pred_std = torch.cat(
                pred_std_list,
                dim=1,
            )
        else:
            pred_std = None

        if "classify" in self.target_task:
            pred_logits = torch.cat(
                pred_logits_list,
                dim=1,
            )
        else:
            pred_logits = None

        return (
            frame_feature,
            pred_mean,
            pred_std,
            pred_logits,
            estimated_flow,
            aligned_ref,
            f_c,
        )

    def forward(
            self,
            x_current,
            x_reference,
            flow=None,
    ):
        (
            _,
            pred_mean,
            pred_std,
            pred_logits,
            estimated_flow,
            aligned_ref,
            f_c,
        ) = self.extract_frame_outputs(
            x_current,
            x_reference,
            flow,
        )

        return (
            pred_mean,
            pred_std,
            pred_logits,
            estimated_flow,
            aligned_ref,
            f_c,
        )

    def forward_sequence(
            self,
            x_current,
            x_reference,
            flow=None,
    ):
        """
        输入：
            x_current:   [B,T,3,H,W]
            x_reference: [B,T,3,H,W]
            flow:        [B,T,2,H,W]

        输出接口仍和原 forward 一致。
        """
        if x_current.dim() != 5:
            raise ValueError(
                f"x_current 应为 [B,T,C,H,W]，"
                f"实际为 {x_current.shape}"
            )

        if x_reference.dim() != 5:
            raise ValueError(
                f"x_reference 应为 [B,T,C,H,W]，"
                f"实际为 {x_reference.shape}"
            )

        B, T, C, H, W = x_current.shape

        current_flat = x_current.reshape(
            B * T,
            C,
            H,
            W,
        )

        reference_flat = x_reference.reshape(
            B * T,
            C,
            H,
            W,
        )

        if flow is not None:
            if flow.dim() != 5:
                raise ValueError(
                    f"flow 应为 [B,T,2,H,W]，"
                    f"实际为 {flow.shape}"
                )

            flow_flat = flow.reshape(
                B * T,
                flow.size(2),
                flow.size(3),
                flow.size(4),
            )
        else:
            flow_flat = None

        (
            frame_feature_flat,
            frame_pred_flat,
            frame_std_flat,
            frame_logits_flat,
            estimated_flow,
            aligned_ref,
            f_c,
        ) = self.extract_frame_outputs(
            current_flat,
            reference_flat,
            flow_flat,
        )

        if frame_pred_flat is None:
            raise RuntimeError(
                "时序模型要求 target_task 中包含 regress"
            )

        frame_features = frame_feature_flat.reshape(
            B,
            T,
            self.frame_feature_dim,
        )

        frame_predictions = frame_pred_flat.reshape(
            B,
            T,
            self.au_number,
        )

        pred_mean = self.temporal_head(
            frame_features,
            frame_predictions,
        )

        center = T // 2

        if frame_std_flat is not None:
            pred_std = frame_std_flat.reshape(
                B,
                T,
                self.au_number,
            )[:, center, :]
        else:
            pred_std = None

        if frame_logits_flat is not None:
            pred_logits = frame_logits_flat.reshape(
                B,
                T,
                -1,
            )[:, center, :]
        else:
            pred_logits = None

        return (
            pred_mean,
            pred_std,
            pred_logits,
            estimated_flow,
            aligned_ref,
            f_c,
        )