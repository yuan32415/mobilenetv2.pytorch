# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .aspp_head import ASPPModule
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@MODELS.register_module()
class Myhead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(Myhead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # APSP Module,Pooling Pyramid Module
        self.dilations = dilations  # 空洞率
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 池化操作（C,1,1）
            ConvModule(  # 卷积操作
                self.in_channels[-1],  # 输入通道,最后一层,768
                self.channels,  # 输出通道512
                1,  # 卷积核大小
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        # 这里获得4个分支
        self.aspp_modules = ASPPModule(
            dilations,  # dilations=(1,6,12,18)
            self.in_channels[-1],  # 输入通道,最后一层,768
            self.channels,  # 输出通道，512
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # 将融合后的特征图转变通道
        self.aspp_bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,  # 一共是5个
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # FPN Module
        self.lateral_convs = nn.ModuleList()  # 横向连接 1*1卷积改变通道数
        self.fpn_convs = nn.ModuleList()  # 纵向连接 上采样2倍，然后3*3卷积消除混叠效应（这里只是3*3卷积）
        for in_channels in self.in_channels[:-1]:  # skip the top layer，in_channels=[256,512,1024,2048]
            l_conv = ConvModule(
                in_channels,  # 输入通道
                self.channels,  # 输出通道512
                1,  # kernal_size,1*1卷积改变通道的维度
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            # 纵向连接
            fpn_conv = ConvModule(
                self.channels,  # 输入通道512
                self.channels,  # 输出通道512
                3,  # kernal_size
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # 融合FPN 4个阶段的输出
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,  # 4*512=2048
            self.channels,  # 输出通道512
            3,  # kernal_size
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def aspp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]  # 最后一个stage的输出
        aspp_outs = [
            resize(
                self.image_pool(x),  # image pooling
                size=x.size()[2:],  # 双线性插值恢复图像分辨率
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)  # 按照通道维度拼接在一起
        feats = self.aspp_bottleneck(aspp_outs)  # 转变通道维度
        return feats

    # 返回：features map（张量）：一个张量（batch_size，self.channels，H，W），是解码器头部最后一层的特征图。
    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # 转变输入multiple_select
        # Multiple feature maps will be bundle into a list and passed into decode head
        inputs = self._transform_inputs(inputs)
        # build
        # 横向连接，1*1卷积改变它们的通道数，（除了最后一个stage）
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # PSP Moudle前向计算
        # self.psp_forward返回的也是一张512通道的feature
        laterals.append(self.aspp_forward(inputs))

        # build top-down path
        # 构建自上而下的通道
        used_backbone_levels = len(laterals)  # len(laterals)=4
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # 双线性插值，然后相加
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        # 进一步完善fpn的输出（因为有混叠效应）
        fpn_outs = [
            self.fpn_convs[i](laterals[i])  # 3*3卷积消除混叠效应
            for i in range(used_backbone_levels - 1)  # 除了psp的输出不用
        ]
        # append psp feature
        # 加上aspp的特征
        fpn_outs.append(laterals[-1])

        # 接下来开始Fuse操作
        # 将增强后每一级特征进行上采样至原图像1/4
        for i in range(used_backbone_levels - 1, 0, -1):  # use_backbone_levels=4
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],  # 1/4尺寸
                mode='bilinear',
                align_corners=self.align_corners)
        # 将FPN的4个输出按照通道维度拼接在一起
        fpn_outs = torch.cat(fpn_outs, dim=1)  # 那就变成了512*4=2048
        # 使用3*3卷积转变为512个通道输出
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        # 生成融合的特征图
        output = self._forward_feature(inputs)
        # 通道数转变为类别数
        output = self.cls_seg(output)
        return output
