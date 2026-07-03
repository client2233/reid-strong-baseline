# encoding: utf-8
"""
@author: reid-strong-baseline
@description: 多种注意力机制模块，用于提升ReID识别性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Non-local Block (自注意力，捕获长距离依赖)
#   参考: "Non-local Neural Networks" (CVPR 2018)
# ============================================================================
class NonLocalBlock(nn.Module):
    """
    非局部块，用于捕获特征图中的长距离空间依赖关系。
    在ReID中，可以更好地建模人体各部分之间的关系。
    """
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', 
                 sub_sample=True, bn_layer=True):
        """
        Args:
            in_channels: 输入通道数
            inter_channels: 中间通道数，默认为 in_channels // 2
            mode: 相似度计算方式，支持 'gaussian', 'dot_product', 'embedded_gaussian'
            sub_sample: 是否对空间维度进行降采样以减少计算量
            bn_layer: 是否使用 BatchNorm
        """
        super(NonLocalBlock, self).__init__()

        assert mode in ['gaussian', 'dot_product', 'embedded_gaussian']

        self.mode = mode
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # theta, phi, g 卷积
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=False)

        # 降采样（可选）
        self.sub_sample = sub_sample
        if sub_sample:
            self.max_pool = nn.MaxPool2d(kernel_size=2)
            self.g_max_pool = nn.MaxPool2d(kernel_size=2)

        # Wz 输出变换
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, bias=False)

        # 可学习缩放参数，初始化为0（从零开始学习）
        nn.init.constant_(self.W[1].weight, 0) if bn_layer else nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W[1].bias, 0) if bn_layer else None

    def forward(self, x):
        batch_size, C, H, W = x.shape

        # theta: [N, C_inter, H, W] -> [N, C_inter, H*W]
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        # phi: [N, C_inter, H, W] -> [N, C_inter, H*W]
        phi_x = self.phi(x)
        if self.sub_sample:
            phi_x = self.max_pool(phi_x)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        # phi_x: [N, C_inter, H2*W2]

        # 相似度矩阵: [N, H*W, H2*W2]
        if self.mode == 'gaussian':
            theta_x = theta_x - theta_x.mean(dim=-1, keepdim=True)
            phi_x = phi_x - phi_x.mean(dim=-1, keepdim=True)
            f = torch.einsum('nci,ncj->nij', theta_x, phi_x)
        elif self.mode == 'dot_product':
            f = torch.einsum('nci,ncj->nij', theta_x, phi_x)
        else:  # embedded_gaussian
            f = torch.einsum('nci,ncj->nij', theta_x, phi_x)
            # 对每个位置做 softmax
            f = f / (C ** 0.5)
            f = F.softmax(f, dim=-1)

        # g: [N, C_inter, H, W] -> [N, C_inter, H*W]
        g_x = self.g(x)
        if self.sub_sample:
            g_x = self.g_max_pool(g_x)
        g_x = g_x.view(batch_size, self.inter_channels, -1)

        # 加权聚合: [N, C_inter, H*W]
        y = torch.einsum('nij,ncj->nci', f, g_x)
        y = y.contiguous()

        # 重塑为 [N, C_inter, H, W]
        y = y.view(batch_size, self.inter_channels, *x.shape[2:])

        # 输出变换
        W_y = self.W(y)
        z = W_y + x  # 残差连接

        return z


# ============================================================================
# 2. CBAM (Convolutional Block Attention Module)
#    参考: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
# ============================================================================
class ChannelAttention(nn.Module):
    """
    通道注意力模块 - 关注"什么"是有意义的特征
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """
    空间注意力模块 - 关注"哪里"是有意义的区域
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    """
    CBAM: 通道注意力 + 空间注意力串联
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================================
# 3. Coordinate Attention (坐标注意力)
#    参考: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
#    在 ReID 中能更好地保留位置信息，提升行人重识别的空间定位能力
# ============================================================================
class CoordAttention(nn.Module):
    """
    坐标注意力模块，同时编码通道关系和方向感知的位置信息。
    相比SE注意力，CA能捕获更精细的位置信息。
    """
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 对宽度池化 -> [C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 对高度池化 -> [C, 1, W]

        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, C, H, W = x.shape

        # 沿着高度和宽度方向分别做全局平均池化
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, 1, W] -> [N, C, W, 1]

        # 拼接后做1x1卷积
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        # 分割回高度和宽度分支
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [N, C, W, 1] -> [N, C, 1, W]

        # 分别生成注意力权重
        att_h = self.sigmoid(self.conv_h(x_h))  # [N, C, H, 1]
        att_w = self.sigmoid(self.conv_w(x_w))  # [N, C, 1, W]

        # 应用注意力
        out = x * att_h * att_w

        return out


# ============================================================================
# 4. External Attention (外部注意力)
#    参考: "Beyond Self-Attention: External Attention" (2020)
#    使用两个外部记忆单元来建模特征，计算量更小
# ============================================================================
class ExternalAttention(nn.Module):
    """
    外部注意力模块，使用可学习的外部记忆单元来替代自注意力中的QK计算。
    计算复杂度 O(N) 而非 O(N^2)。
    """
    def __init__(self, in_channels, S=64, d=8):
        """
        Args:
            in_channels: 输入通道数
            S: 外部记忆单元的大小 (memory size)
            d: 特征降维维度
        """
        super(ExternalAttention, self).__init__()
        self.in_channels = in_channels
        self.S = S
        self.d = d

        # 两个线性变换用于降维和升维
        self.project = nn.Linear(in_channels, d, bias=False)
        self.M_k = nn.Linear(d, S, bias=False)  # 外部记忆 Key
        self.M_v = nn.Linear(S, d, bias=False)  # 外部记忆 Value
        self.softmax = nn.Softmax(dim=-1)
        self.project_back = nn.Linear(d, in_channels, bias=False)

        # LayerNorm 稳定训练
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        # 残差
        identity = x_flat

        # 投影降维
        x_proj = self.project(x_flat)  # [B, N, d]

        # 外部注意力计算
        attn = self.M_k(x_proj)  # [B, N, S]
        attn = self.softmax(attn)
        attn = attn / (1e-6 + attn.sum(dim=1, keepdim=True))  # 归一化

        out = self.M_v(attn)  # [B, N, d]
        out = self.project_back(out)  # [B, N, C]

        # 残差连接 + LayerNorm
        out = self.norm(out + identity)

        # 重塑回 [B, C, H, W]
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out


# ============================================================================
# 5. Gather-Excite (Gather-Excite Attention)
#    参考: "Gather-Excite: Bringing Attention to CNNs" (NeurIPS 2019)
#    通过更丰富的上下文聚合来增强特征表达能力
# ============================================================================
class GatherExcite(nn.Module):
    """
    Gather-Excite 注意力模块
    Gather: 用深度可分离卷积聚合更大范围的上下文
    Excite: 用1x1卷积 + sigmoid 生成通道注意力
    """
    def __init__(self, in_channels, reduction=16, extent=0):
        """
        Args:
            in_channels: 输入通道数
            reduction: 压缩比例
            extent: 聚合范围，0表示全局聚合，>0表示局部聚合使用的卷积核大小
        """
        super(GatherExcite, self).__init__()
        self.extent = extent

        if extent == 0:
            # 全局聚合
            self.gather = nn.AdaptiveAvgPool2d(1)
        else:
            # 局部聚合 - 使用深度可分离卷积
            self.gather = nn.Conv2d(
                in_channels, in_channels, kernel_size=extent, 
                stride=1, padding=extent//2, groups=in_channels, bias=False
            )

        # Excite 模块
        mid_channels = max(8, in_channels // reduction)
        self.excite = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.extent == 0:
            g = self.gather(x)
        else:
            g = self.gather(x)
            g = F.adaptive_avg_pool2d(g, 1)
        
        att = self.excite(g)
        return x * att


# ============================================================================
# 6. Transformer Encoder Block (标准Transformer编码器块)
#    参考: "Attention Is All You Need" (NeurIPS 2017)
#    MHSA (Multi-Head Self-Attention) + FFN (Feed-Forward Network)
#    作为可插拔模块与CNN混合使用，捕获全局依赖关系
# ============================================================================
class TransformerEncoderBlock(nn.Module):
    """
    标准 Transformer Encoder 块（MHSA + FFN），可直接插入 CNN 之后。
    
    接收 4D 特征图 [B, C, H, W] → 展平为序列 → 3D [B, N, C] →
    LayerNorm → MHSA → Residual → LayerNorm → MLP → Residual →
    重塑回 4D [B, C, H, W]
    
    相比 Non-local block，Transformer 使用:
    - LayerNorm 而非 BN
    - 多头注意力 (Multi-Head) 捕获更丰富的特征
    - 更深的 FFN (MLP) 增强表达能力
    - 没有显式的残差缩放 (恒等映射)
    """
    def __init__(self, in_channels, num_heads=8, mlp_ratio=4., dropout=0.1, 
                 use_pe=True, max_h=24, max_w=8):
        """
        Args:
            in_channels: 输入通道数 (= embedding dim)
            num_heads: 多头注意力的头数
            mlp_ratio: MLP 隐藏层维度 = in_channels * mlp_ratio
            dropout: Dropout 概率
            use_pe: 是否使用可学习的位置编码
            max_h, max_w: 最大空间尺寸，用于位置编码
        """
        super(TransformerEncoderBlock, self).__init__()
        
        embed_dim = in_channels
        self.embed_dim = embed_dim
        self.use_pe = use_pe
        
        # 可学习位置编码 (1D 序列位置编码)
        if use_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_h * max_w, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # LayerNorm (Pre-Norm 结构，更稳定)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 多头自注意力
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Feed-Forward Network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征图
        Returns:
            [B, C, H, W] 与输入同形状
        """
        B, C, H, W = x.shape
        N = H * W
        
        # Flatten: [B, C, H, W] -> [B, N, C]
        x_flat = x.view(B, C, N).permute(0, 2, 1).contiguous()  # [B, N, C]
        
        # 加入位置编码
        if self.use_pe:
            if N <= self.pos_embed.shape[1]:
                x_flat = x_flat + self.pos_embed[:, :N, :]
            else:
                # 插值位置编码 (双线性插值)
                pe = self.pos_embed.permute(0, 2, 1).view(1, C, self.pos_embed.shape[1])
                pe = F.interpolate(pe, size=N, mode='linear', align_corners=False)
                pe = pe.view(1, C, N).permute(0, 2, 1).contiguous()
                x_flat = x_flat + pe
        
        # Pre-Norm + MHSA + Residual
        shortcut = x_flat
        x_norm = self.norm1(x_flat)
        
        # QKV 投影
        qkv = self.qkv(x_norm)  # [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        # 残差连接 1
        x_flat = shortcut + x_attn
        
        # Pre-Norm + MLP + Residual
        shortcut = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x_flat = shortcut + x_flat
        
        # 重塑回 4D: [B, N, C] -> [B, C, H, W]
        out = x_flat.permute(0, 2, 1).view(B, C, H, W).contiguous()
        
        return out


# ============================================================================
# 7. 注意力插件 — 可插入到ResNet各层之后
# ============================================================================
ATTENTION_REGISTRY = {
    'non_local': NonLocalBlock,
    'cbam': CBAM,
    'coord_att': CoordAttention,
    'external_att': ExternalAttention,
    'gather_excite': GatherExcite,
    'transformer': TransformerEncoderBlock,
}


def build_attention(att_type, in_channels, **kwargs):
    """
    根据类型构建注意力模块
    
    Args:
        att_type (str): 注意力类型
        in_channels (int): 输入通道数
        **kwargs: 注意力模块额外参数
    
    Returns:
        nn.Module: 注意力模块，如果att_type为'none'则返回nn.Identity()
    """
    if att_type in ATTENTION_REGISTRY:
        att_class = ATTENTION_REGISTRY[att_type]
        return att_class(in_channels, **kwargs)
    else:
        return nn.Identity()
