# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch

#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, channels,reduction_radio=16): #输入通道数，通道压缩量
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels  // reduction_radio
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1)) #最大池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #平均实话
        self.mlp = nn.Sequential(  # 使用1x1卷积代替线性层，可以不用调整tensor的形状
            nn.Conv2d(self.channels, self.inter_channels,
                    kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(self.inter_channels), #有这个效果极差！！！
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.channels,
                    kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(self.channels) #有这个效果极差！！
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        maxout = self.maxpool(x) # (b, c, 1, 1)
        avgout = self.avgpool(x) # (b, c, 1, 1)
        maxout = self.mlp(maxout) # (b, c, 1, 1)
        avgout = self.mlp(avgout) # (b, c, 1, 1)
        attention = self.sigmoid(maxout + avgout) #(b, c, 1, 1)
        return attention

#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1,
                    kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        maxpool = x.argmax(dim=1, keepdim=True)  # (b, 1, h, w)
        avgpool = x.mean(dim=1, keepdim=True)  # (b, 1, h, w)

        out = torch.cat([maxpool, avgpool], dim=1)  # (b, 2, h, w)
        out = self.conv(out)  # (b, 1, h, w)
        attention = self.sigmoid(out)  # (b, 1, h, w)
        return attention

class CBAM(nn.Module):
    def __init__(self,in_channels):
        super(CBAM, self).__init__()
        self.in_channels=in_channels
        self.channel_atten=ChannelAttention(self.in_channels)
        self.spatial_atten=SpatialAttention()

    def forward(self, x):
        channel_atten = self.channel_atten(x)
        x = channel_atten * x

        spatial_atten = self.spatial_atten(x)
        x = spatial_atten * x
        return x

