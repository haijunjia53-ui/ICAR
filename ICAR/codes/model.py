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
        # 【还原】0.61版本没有冻结层

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim=1024, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4), nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, x_curr, x_ref):
        b, c, h, w = x_curr.shape
        curr_flat = x_curr.flatten(2).permute(2, 0, 1)
        ref_flat = x_ref.flatten(2).permute(2, 0, 1)
        q = self.q_proj(curr_flat)
        k = self.k_proj(ref_flat)
        v = self.v_proj(ref_flat)
        attn_output, _ = self.cross_attention(q, k, v)
        x = self.norm1(q + attn_output)
        x = self.norm2(x + self.ffn(x))
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x


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
        # 【还原】0.61版本用的是3 Patch的初始化
        path_postion = [1, 0, 0, 0, 1 / 3, -13 / 20,
                        3 / 4, 0, 0, 0, 1 / 4, -1 / 10,
                        1, 0, 0, 0, 1 / 3, 5 / 10]
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(path_postion, dtype=torch.float))

    def forward(self, x):
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

        self.encoder = ResNetFeatureExtractor(pretrained=True)
        self.backbone_out_channels = 1024
        self.alignment = ImplicitMotionAlignment(feature_dim=self.backbone_out_channels)

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

    def forward(self, x_current, x_reference):
        f_c = self.encoder(x_current)
        f_r = self.encoder(x_reference)
        aligned_ref = self.alignment(f_c, f_r)
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
            # 【还原】0.61版本中已开启 ReLU (修复了负值预测)
            pred_mean = torch.relu(pred_mean)
        if 'uncertain' in self.target_task:
            pred_std = torch.cat(pred_std, dim=1)
        if 'classify' in self.target_task:
            pred_logits = torch.cat(pred_logits, dim=1)

        return pred_mean, pred_std, pred_logits