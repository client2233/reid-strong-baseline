# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os

import torch
import torchvision
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.vit import build_vit_backbone, ViTBackbone


# 不同骨干网络对应的 torchvision 预训练权重下载 URL
PRETRAINED_URLS = {
    'resnet18': torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
    'resnet34': torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
    'resnet50': torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
}


def _ensure_pretrained_weight(model_name, model_path):
    """
    确保预训练权重文件存在。
    
    对已知的骨干网络，忽略传入的 model_path，按统一命名规则管理权重：
      .torch/models/{model_name}-imagenet.pth
    这样即使 lightweight.yml 指定了错误的路径，也能自动下载正确的权重。
    """
    if model_name not in PRETRAINED_URLS:
        return model_path

    # 使用标准路径，不受配置文件干扰
    # 权重统一保存到 .torch/models/{model_name}-imagenet.pth
    if model_path:
        cache_dir = os.path.dirname(os.path.abspath(model_path))
    else:
        cache_dir = os.path.abspath(".torch/models")
    std_path = os.path.join(cache_dir, "{}-imagenet.pth".format(model_name))
    
    if not os.path.exists(std_path):
        os.makedirs(cache_dir, exist_ok=True)
        print("  Auto-downloading {} pretrained weights from torchvision...".format(model_name))
        weights_enum = PRETRAINED_URLS[model_name]
        full_model = getattr(torchvision.models, model_name)(weights=weights_enum)
        torch.save(full_model.state_dict(), std_path)
        print("  Saved pretrained weights to {}".format(std_path))
        del full_model

    return std_path


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,
                 att_type=None, att_pos=None,
                 img_size_h=384, img_size_w=128):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2],
                               att_type=att_type, att_pos=att_pos)
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3],
                               att_type=att_type, att_pos=att_pos)
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               att_type=att_type, att_pos=att_pos)
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3],
                               att_type=att_type, att_pos=att_pos)
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3],
                               att_type=att_type, att_pos=att_pos)
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        elif model_name in ['vit_tiny_patch16', 'vit_small_patch16', 'vit_base_patch16',
                            'vit_base_patch32', 'vit_large_patch16', 'vit_large_patch32']:
            self.base = build_vit_backbone(
                model_name=model_name,
                img_size_h=img_size_h,
                img_size_w=img_size_w,
                pretrain_choice=pretrain_choice,
            )
            # ViT 的 in_planes 由 embed_dim 决定
            # 如果是 vit_base_patch16: 768, vit_large_patch16: 1024
            # 从 base 中获取 embed_dim
            if hasattr(self.base, 'embed_dim'):
                self.in_planes = self.base.embed_dim

        if pretrain_choice == 'imagenet':
            # ViT 在 build_vit_backbone 内部已加载预训练权重，跳过避免覆盖
            if model_name not in ['vit_tiny_patch16', 'vit_small_patch16', 'vit_base_patch16',
                                  'vit_base_patch32', 'vit_large_patch16', 'vit_large_patch32']:
                # 确保预训练权重存在（自动下载）
                model_path = _ensure_pretrained_weight(model_name, model_path)
                self.base.load_param(model_path)
                print('Loading pretrained ImageNet model from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
