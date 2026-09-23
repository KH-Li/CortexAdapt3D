import torch
import torch.nn as nn
import math

from timm.models.layers import DropPath, trunc_normal_

from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from models.spectral_adapter import SpectralAdapter, get_basis
import numpy as np

class RelPosBias(nn.Module):
    """
    Relative position bias from pairwise relative xyz (optionally + distance).
    Output: [B, num_heads, G, G]
    """
    def __init__(self, num_heads: int, hidden: int = 128, use_dist: bool = True):
        super().__init__()
        self.use_dist = use_dist
        in_dim = 4 if use_dist else 3  # dx,dy,dz,(dist)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_heads)
        )

    def forward(self, center: torch.Tensor) -> torch.Tensor:
        """
        center: [B, G, 3]
        return: [B, H, G, G]
        """
        rel = center[:, :, None, :] - center[:, None, :, :]  # [B,G,G,3]
        if self.use_dist:
            dist = torch.norm(rel, dim=-1, keepdim=True)     # [B,G,G,1]
            rel_feat = torch.cat([rel, dist], dim=-1)        # [B,G,G,4]
        else:
            rel_feat = rel                                   # [B,G,G,3]

        bias = self.mlp(rel_feat)                            # [B,G,G,H]
        return bias.permute(0, 3, 1, 2).contiguous()          # [B,H,G,G]

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias: torch.Tensor = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if rel_pos_bias is not None:                          
            # rel_pos_bias: [B,H,N,N]
            attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, cfg, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapt=True):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.adapt = adapt
        if adapt:
            self.spectral_adapter = SpectralAdapter(dim, cfg)

    def forward(self, x, attr, U, rel_pos_bias=None):
        x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias))
        if self.adapt:
            t = self.spectral_adapter(x, attr, U)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.adapt:
            x = x + t
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                cfg=cfg,
            )
            for i in range(depth)])

    def forward(self, x, pos, attr, rel_pos_bias, U):
        for _, block in enumerate(self.blocks):
            x = block(x + pos, attr, U, rel_pos_bias=rel_pos_bias)
        return x

class Patch_divider(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, attr, split_index, center_index):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        split_index = torch.as_tensor(split_index, device=xyz.device, dtype=torch.long)
        center_index = torch.as_tensor(center_index, device=xyz.device, dtype=torch.long)
        B, num_points, _ = xyz.shape
        # fps the centers out
        expanded_split_index = split_index.unsqueeze(0).repeat(B, 1, 1)
        expanded_center_index = center_index.unsqueeze(0).repeat(B, 1, 1)
        batch_indices = torch.arange(B, device=xyz.device).view(B, 1, 1)
        neighborhood = xyz[batch_indices, expanded_split_index]
        center_xyz = xyz[batch_indices, expanded_center_index]
        center = torch.mean(center_xyz, dim=2)
        neighborhood = neighborhood - center.unsqueeze(2)

        neighborhood_attr = attr[batch_indices, expanded_split_index]
        center_attr = attr[batch_indices, expanded_center_index]    
        center_attr = torch.mean(center_attr, dim=2)
        neighborhood_attr = neighborhood_attr - center_attr.unsqueeze(2)

        return neighborhood, neighborhood_attr, center

@MODELS.register_module()
class CortexAdapt3D(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.split_index = np.load(config.split_index_path)
        self.center_index = np.load(config.center_index_path)

        self.patch_divider = Patch_divider(num_group = self.num_group, group_size = self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.attr_encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            cfg=config,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.rel_pos_bias = RelPosBias(num_heads=self.num_heads, hidden=128, use_dist=True)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, map_location="cpu", weights_only=False)
            #base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['pointdif'].items()}
            base_ckpt = {k.removeprefix("module.").replace("gft_adapter.", "spectral_adapter."): v
                         for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('mask_encoder'):
                    base_ckpt[k[len('mask_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                if k.startswith('ACT_encoder'):
                    base_ckpt[k[len('ACT_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, attr):

        neighborhood, neighborhood_attr, center = self.patch_divider(pts, attr, self.split_index, self.center_index)

        U = get_basis(center)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        group_attr_tokens = self.attr_encoder(neighborhood_attr)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        rel_pos_bias = self.rel_pos_bias(center)

        B, H, G, _ = rel_pos_bias.shape
        rel_pos_bias_full = rel_pos_bias.new_zeros(B, H, G + 1, G + 1)
        rel_pos_bias_full[:, :, 1:, 1:] = rel_pos_bias

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)

        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos, group_attr_tokens, rel_pos_bias_full, U)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
