import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_resnet50_no_weights():
    """Build the same torchvision ResNet-50 architecture without ImageNet weights."""
    try:
        return models.resnet50(weights=None)
    except TypeError:  # torchvision < 0.13
        return models.resnet50(pretrained=False)


def _safe_torch_load(path):
    """Load the official AffectNet checkpoint; PyTorch 2.6 compatible."""
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print('[R8-AffectAU] weights_only=True failed:', repr(e))
        print('[R8-AffectAU] Retrying with weights_only=False; ONLY use a trusted official checkpoint.')
        return torch.load(path, map_location='cpu', weights_only=False)


def load_affectnet_au_guided_resnet50(resnet, checkpoint_path):
    """
    Load ONLY the ResNet-50 encoder weights from the official
    AffectNet / ResNet50 / CAM / align_atten_to_heatmap_True checkpoint.

    The checkpoint observed from the official release has top-level keys:
        encoder              -> 318 ResNet tensors (no fc)
        classification_head  -> 7-class AffectNet head

    ICAR uses only conv1/bn1/layer1/layer2/layer3.  We require 100% shape
    coverage for those layers and intentionally ignore the AffectNet head
    and layer4 (ICAR truncates before layer4).
    """
    import os

    if checkpoint_path is None or str(checkpoint_path).strip() == '':
        raise ValueError('--affect_pretrained must point to the official AffectNet model.pt')
    checkpoint_path = os.path.expanduser(str(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('AffectNet checkpoint not found: {}'.format(checkpoint_path))

    ckpt = _safe_torch_load(checkpoint_path)
    if not isinstance(ckpt, dict):
        raise TypeError('Checkpoint must be a dict, got {}'.format(type(ckpt)))
    if 'encoder' not in ckpt or not isinstance(ckpt['encoder'], dict):
        raise KeyError('Expected top-level key "encoder". Got keys: {}'.format(list(ckpt.keys())[:30]))

    raw_state = ckpt['encoder']
    own_state = resnet.state_dict()
    critical_prefixes = ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.')

    converted = {}
    shape_mismatch = []
    unknown_critical = []
    for key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        key = str(key)
        if not key.startswith(critical_prefixes):
            continue
        if key not in own_state:
            unknown_critical.append(key)
            continue
        if tuple(value.shape) != tuple(own_state[key].shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(own_state[key].shape)))
            continue
        converted[key] = value

    expected_critical = [
        k for k in own_state.keys()
        if k.startswith(critical_prefixes) and not k.endswith('num_batches_tracked')
    ]
    loaded_critical = [k for k in expected_critical if k in converted]
    missing_critical = [k for k in expected_critical if k not in converted]

    raw_tensor_count = sum(torch.is_tensor(v) for v in raw_state.values())
    head_tensor_count = 0
    if isinstance(ckpt.get('classification_head', None), dict):
        head_tensor_count = sum(torch.is_tensor(v) for v in ckpt['classification_head'].values())

    print('==========================================================')
    print('[R8-AffectAU] Checkpoint:', checkpoint_path)
    print('[R8-AffectAU] Top-level keys:', list(ckpt.keys()))
    print('[R8-AffectAU] Encoder tensor keys:', raw_tensor_count)
    print('[R8-AffectAU] Classification-head tensors (IGNORED):', head_tensor_count)
    print('[R8-AffectAU] ICAR layer0-layer3 coverage: {}/{} ({:.1f}%)'.format(
        len(loaded_critical), len(expected_critical),
        100.0 * len(loaded_critical) / max(1, len(expected_critical))))
    print('[R8-AffectAU] Shape mismatches:', len(shape_mismatch))
    print('[R8-AffectAU] Unknown critical keys:', len(unknown_critical))

    if shape_mismatch:
        print('[R8-AffectAU] Example shape mismatch:', shape_mismatch[:5])
    if unknown_critical:
        print('[R8-AffectAU] Example unknown critical key:', unknown_critical[:5])
    if missing_critical:
        print('[R8-AffectAU] Missing critical keys (first 20):', missing_critical[:20])
        raise RuntimeError(
            'AffectNet checkpoint is not fully compatible with ICAR torchvision ResNet-50. '
            'Do NOT train unless layer0-layer3 coverage is 100%.'
        )

    # Load only layers actually used by ICAR.  strict=False is intentional because
    # layer4/fc are irrelevant to the truncated encoder.
    msg = resnet.load_state_dict(converted, strict=False)
    post_missing = [
        k for k in msg.missing_keys
        if k.startswith(critical_prefixes) and not k.endswith('num_batches_tracked')
    ]
    if post_missing:
        raise RuntimeError('Critical keys missing after load: {}'.format(post_missing[:20]))

    conv1 = resnet.conv1.weight.detach().float()
    print('[R8-AffectAU] conv1 fingerprint: mean={:.6f}, std={:.6f}, absmax={:.6f}'.format(
        conv1.mean().item(), conv1.std().item(), conv1.abs().max().item()))
    print('[R8-AffectAU] PASS: full ICAR encoder (through layer3) initialized from AffectNet AU-guided ResNet50.')
    print('==========================================================')

    return {
        'critical_loaded': len(loaded_critical),
        'critical_expected': len(expected_critical),
        'shape_mismatch': shape_mismatch,
        'unknown_critical': unknown_critical,
        'encoder_tensor_count': raw_tensor_count,
        'head_tensor_count': head_tensor_count,
    }


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, affect_ckpt=None):
        super().__init__()
        if affect_ckpt is not None and str(affect_ckpt).strip() != '':
            # Same torchvision ResNet50 architecture as baseline; only initialization changes.
            resnet = _make_resnet50_no_weights()
            load_affectnet_au_guided_resnet50(resnet, affect_ckpt)
        else:
            # Baseline fallback for diagnostics only.
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                resnet = models.resnet50(weights=weights)
            except (AttributeError, TypeError):
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


class AU_IMF_Net(nn.Module):
    def __init__(self, opts):
        super(AU_IMF_Net, self).__init__()

        self.encoder = ResNetFeatureExtractor(pretrained=False, affect_ckpt=getattr(opts, 'affect_pretrained', None))
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
        patch_au_num = [5, 1, 6]

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

    def forward(self, x_current, x_reference, flow=None):
        f_c = self.encoder(x_current)
        f_r = self.encoder(x_reference)

        estimated_flow = None

        if self.align_mode == 'implicit':
            # 隐式学习：网络自己预测 flow 并对齐
            # 注意：这里调用的是 alignment (LearnableFeatureWarping)
            aligned_ref, estimated_flow = self.alignment(f_c, f_r)

        elif self.align_mode == 'flow':
            # 显式光流：使用传入的 OpenCV 光流
            assert flow is not None, "align_mode=flow requires flow input"
            _, _, h, w = f_c.shape
            flow_h, flow_w_ = flow.shape[-2], flow.shape[-1]

            # 将光流图 downsample 到特征图大小
            flow_feat = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
            # 缩放位移值
            flow_feat[:, 0, :, :] *= (w / float(flow_w_))
            flow_feat[:, 1, :, :] *= (h / float(flow_h))

            aligned_ref = flow_warp(f_r, flow_feat)
        else:
            # None: 不对齐
            aligned_ref = f_r

        # 计算差分特征
        motion_feature = f_c - aligned_ref

        patch = self.patch_proposal(motion_feature)

        pred_mean = []
        pred_std = []
        pred_logits = []
        patch_feature = []

        for i in range(self.patch_number):
            feature = F.adaptive_avg_pool2d(patch[i], 1)
            feature = self.new.down[i](feature)
            feature = torch.flatten(feature, 1)
            patch_feature.append(feature)

        patch_feature = self.self_attention(patch_feature)

        for i in range(self.patch_number):
            if 'regress' in self.target_task:
                pred_mean.append(self.new.regress[i](patch_feature[i]))
            if 'uncertain' in self.target_task:
                pred_std.append(self.new.uncertain[i](patch_feature[i]))
            if 'classify' in self.target_task:
                pred_logits.append(self.new.classify[i](patch_feature[i]))

        if 'regress' in self.target_task:
            pred_mean = torch.cat(pred_mean, dim=1)
            pred_mean = torch.relu(pred_mean)
        if 'uncertain' in self.target_task:
            pred_std = torch.cat(pred_std, dim=1)
        if 'classify' in self.target_task:
            pred_logits = torch.cat(pred_logits, dim=1)

        # 返回额外信息用于 Loss 计算: estimated_flow, aligned_ref, f_c
        return pred_mean, pred_std, pred_logits, estimated_flow, aligned_ref, f_c