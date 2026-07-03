# encoding: utf-8
"""
@author: reid-strong-baseline
@description: Vision Transformer (ViT) 骨干网络，用于行人重识别
    参考:
        - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)
        - "TransReID: Transformer-based Object Re-Identification" (ICCV 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ============================================================================
# Patch Embedding: 将图像分割为 patches 并映射到 embedding space
# ============================================================================
class PatchEmbed(nn.Module):
    """
    图像到 Patch 序列的嵌入层
    输入: [B, 3, H, W]
    输出: [B, N, D] 其中 N = (H/P) * (W/P)
    """
    def __init__(self, img_size_h=384, img_size_w=128, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # 使用卷积实现 Patch Embedding (高效)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 支持动态输入尺寸（只要能被 patch_size 整除即可）
        # 位置编码不匹配时会在 ViTBackbone.forward 中自动插值
        
        # [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ============================================================================
# Transformer Encoder Layer
# ============================================================================
class TransformerEncoderLayer(nn.Module):
    """
    标准 Transformer Encoder 层
    Pre-Norm 结构: LayerNorm -> MHSA -> Residual -> LayerNorm -> FFN -> Residual
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1, attn_dropout=0.):
        super().__init__()
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 多头自注意力 (MHSA)
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Feed-Forward Network (MLP)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def _forward_attention(self, x):
        B, N, C = x.shape
        
        # QKV 投影
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权聚合
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def forward(self, x):
        # Pre-Norm + MHSA + Residual
        x = x + self._forward_attention(self.norm1(x))
        # Pre-Norm + MLP + Residual
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Vision Transformer Backbone - 专为 ReID 设计
# ============================================================================
class ViTBackbone(nn.Module):
    """
    Vision Transformer 骨干网络
    
    特点:
    - 与 Baseline 类兼容，输出 [B, D, H', W'] 形状的特征图
    - 支持 [CLS] token 和 patch token 两种输出方式
    - 可加载 ImageNet 预训练权重
    - 专为 ReID 的行人图像比例 (H>W) 优化
    
    默认配置 (ViT-B/16):
        - embed_dim: 768
        - depth: 12
        - num_heads: 12
        - patch_size: 16
        - mlp_ratio: 4.0
    """
    def __init__(
        self, 
        img_size_h=384,
        img_size_w=128,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        dropout=0.1,
        attn_dropout=0.,
        use_cls_token=True,
        output_feat='patch',  # 'cls', 'patch', 'both'
    ):
        super().__init__()
        
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.output_feat = output_feat
        
        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size_h, img_size_w, patch_size, 3, embed_dim
        )
        
        # [CLS] token (可学习)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            num_tokens = self.num_patches + 1
        else:
            num_tokens = self.num_patches
        
        # 位置编码 (可学习 1D 位置编码)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 位置编码 Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        
        # 最终 LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # [CLS] token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]
        
        # 加入位置编码
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            # 处理尺寸不匹配 (例如微调时输入尺寸变化)
            pe = self.pos_embed
            if x.shape[1] < pe.shape[1]:
                x = x + pe[:, :x.shape[1], :]
            else:
                # 对位置编码进行插值
                pe = pe.permute(0, 2, 1)  # [1, D, N_pe]
                pe = F.interpolate(pe, size=x.shape[1], mode='linear', align_corners=False)
                pe = pe.permute(0, 2, 1)  # [1, N_new, D]
                x = x + pe
        
        x = self.pos_drop(x)
        
        # Transformer Encoder
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # 根据输出模式返回特征
        if self.output_feat == 'cls':
            # 仅返回 [CLS] token (需与 GAP 兼容，reshape 为 2D)
            cls_out = x[:, 0]  # [B, D]
            return cls_out.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        
        elif self.output_feat == 'both':
            # 返回 [CLS] + patch tokens 拼接
            cls_out = x[:, 0]  # [B, D]
            patch_out = x[:, 1:]  # [B, N, D]
            # 重塑 patch tokens 为 4D 特征图
            patch_out = patch_out.permute(0, 2, 1)  # [B, D, N]
            patch_out = patch_out.view(B, self.embed_dim, self.num_patches_h, self.num_patches_w)
            return patch_out  # [B, D, H', W']
        
        else:  # 'patch'
            # 仅返回 patch tokens
            if self.use_cls_token:
                x = x[:, 1:]  # 移除 [CLS] token
            # 重塑为 4D 特征图 [B, D, H', W']
            x = x.permute(0, 2, 1)  # [B, D, N]
            x = x.view(B, self.embed_dim, self.num_patches_h, self.num_patches_w)
            return x
    
    def load_param(self, model_path):
        """加载预训练权重，兼容 torchvision ViT 格式"""
        param_dict = torch.load(model_path, map_location='cpu')
        
        # 处理 state_dict 嵌套
        if 'model' in param_dict:
            param_dict = param_dict['model']
        elif 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        model_state = self.state_dict()
        
        # 去除分类头的权重
        loaded_keys = []
        for k, v in param_dict.items():
            # 跳过分类头
            if 'head' in k or 'classifier' in k or 'fc' in k:
                continue
            # 跳过预训练位置编码的尺寸不匹配
            if 'pos_embed' in k and v.shape != model_state[k].shape:
                print(f"  Skip {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
                continue
            # 跳过 cls_token 的尺寸不匹配
            if 'cls_token' in k and v.shape != model_state[k].shape:
                print(f"  Skip {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
                continue
            
            if k in model_state:
                try:
                    model_state[k].copy_(v)
                    loaded_keys.append(k)
                except Exception as e:
                    print(f"  Skip {k}: {e}")
            else:
                print(f"  Skip {k}: not in model")
        
        print(f"  Loaded {len(loaded_keys)}/{len(model_state)} params from {model_path}")
    
    def random_init(self):
        """随机初始化（无预训练时使用）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ============================================================================
# 预训练权重 URL (来自 timm 库)
# ============================================================================
VIT_PRETRAINED_URLS = {
    'vit_base_patch16_224': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
    'vit_base_patch32_224': 'https://download.pytorch.org/models/vit_b_32-c86f1d30.pth',
    'vit_large_patch16_224': 'https://download.pytorch.org/models/vit_l_16-8524937a.pth',
    'vit_large_patch32_224': 'https://download.pytorch.org/models/vit_l_32-9b72c061.pth',
}


def _load_vit_pretrained_weights(model, model_name):
    """加载 ViT ImageNet 预训练权重"""
    if model_name in VIT_PRETRAINED_URLS:
        url = VIT_PRETRAINED_URLS[model_name]
        print(f"  Loading pretrained ViT weights from torchvision...")
        try:
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            model.load_param(state_dict)
        except Exception as e:
            print(f"  Failed to load pretrained weights: {e}")
            print(f"  Using random initialization instead.")
    else:
        print(f"  No pretrained URL for {model_name}, using random init.")


# ============================================================================
# 工厂函数：构建不同规模的 ViT
# ============================================================================
def build_vit_backbone(model_name, img_size_h=384, img_size_w=128, pretrain_choice='imagenet'):
    """
    构建 ViT 骨干网络
    
    Args:
        model_name: 'vit_base_patch16', 'vit_base_patch32', 'vit_small_patch16', 'vit_large_patch16'
        img_size_h: 输入图像高度
        img_size_w: 输入图像宽度
        pretrain_choice: 'imagenet' 或 'self'
    
    Returns:
        ViTBackbone 实例
    """
    configs = {
        'vit_tiny_patch16': {
            'embed_dim': 192, 'depth': 12, 'num_heads': 3, 'patch_size': 16, 'mlp_ratio': 4.0,
        },
        'vit_small_patch16': {
            'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'patch_size': 16, 'mlp_ratio': 4.0,
        },
        'vit_base_patch16': {
            'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'patch_size': 16, 'mlp_ratio': 4.0,
        },
        'vit_base_patch32': {
            'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'patch_size': 32, 'mlp_ratio': 4.0,
        },
        'vit_large_patch16': {
            'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'patch_size': 16, 'mlp_ratio': 4.0,
        },
        'vit_large_patch32': {
            'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'patch_size': 32, 'mlp_ratio': 4.0,
        },
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown ViT model: {model_name}. Options: {list(configs.keys())}")
    
    cfg = configs[model_name]
    
    model = ViTBackbone(
        img_size_h=img_size_h,
        img_size_w=img_size_w,
        patch_size=cfg['patch_size'],
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        dropout=0.1,
        attn_dropout=0.,
        use_cls_token=True,
        output_feat='patch',  # 输出 patch 特征图以兼容 GAP
    )
    
    if pretrain_choice == 'imagenet':
        _load_vit_pretrained_weights(model, model_name)
    
    return model
