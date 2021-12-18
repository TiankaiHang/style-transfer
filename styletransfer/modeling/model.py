from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class StyleTransfer(nn.Module):
    def __init__(self, in_channel=3, num_resblocks=5) -> None:
        super().__init__()
        self.down1 = conv9x9_downsample(in_planes=in_channel, out_planes=64)
        self.down2 = conv3x3(64, 128, 2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        self.res = nn.Sequential(
            *[
                BasicBlock(128, 128) for _ in range(num_resblocks)
            ]
        )

        self.up1 = conv3x3_upsample(128, 64)
        self.up2 = conv9x9_upsample(64, 3)

        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        _, _, H, W = x.shape

        x = F.relu(self.bn1(self.down1(x)))
        x = F.relu(self.bn2(self.down2(x)))

        x = self.res(x)

        x = F.relu(self.bn3(self.up1(x)))
        x = self.up2(x)

        if x.shape[2] != W or x.shape[3] != H:
            x = F.interpolate(x, size=(H, W), mode='bicubic')

        x = (x - x.min()) / (x.max() - x.min())
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv9x9_downsample(in_planes: int, out_planes: int, stride: int = 2) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=9, 
                     padding=4, stride=stride, bias=False)
    

def conv9x9_upsample(in_planes: int, out_planes: int, stride: int = 2) -> nn.Conv2d:
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=9, 
                     padding=3, stride=stride, bias=False)

def conv3x3_upsample(in_planes: int, out_planes: int, stride: int = 2) -> nn.Conv2d:
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, 
                     padding=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256).cuda()
    # model = conv9x9_upsample(3, 64).cuda()
    model = StyleTransfer().cuda()

    y = model(x)

    print(f"x.shape {x.shape}, y.shape {y.shape}")