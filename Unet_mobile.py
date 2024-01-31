import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# conv_bn为网络的第一个卷积块，步长为2
def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


# conv_dw为深度可分离卷积
def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        # 3x3卷积提取特征，步长为2
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # 1x1卷积，步长为1
        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


class mobilenet(nn.Module):
    def __init__(self, n_channels):
        super(mobilenet, self).__init__()
        self.model = MobileNet(n_channels)

    def forward(self, x):
        out3 = self.model.layer1(x)
        out4 = self.model.layer2(out3)
        out5 = self.model.layer3(out4)

        return out3, out4, out5


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Mobile_UNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(Mobile_UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes

        # ---------------------------------------------------#
        #   48,48,128；24,24,512；16,16,1024
        # ---------------------------------------------------#
        self.backbone = mobilenet(n_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = DoubleConv(128, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = DoubleConv(128, 32)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = DoubleConv(64, 24)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        #nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = DoubleConv(48, 16)

        self.oup = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)
        # print(f"x2.shape: {x2.shape}, x1: {x1.shape}, x0: {x0.shape} ")

        P5 = self.up1(x0)
        P5 = self.conv1(P5)           # P5: 26x26x512
        # print(P5.shape)
        P4 = x1                       # P4: 26x26x512
        P4 = torch.cat([P4, P5], axis=1)   # P4(堆叠后): 26x26x1024
        # print(f"cat 后是： {P4.shape}")

        P4 = self.up2(P4)             # 52x52x1024
        P4 = self.conv2(P4)           # 52x52x192
        P3 = x2                       # x2 = 52x52x256
        P3 = torch.cat([P4, P3], axis=1)  # 52x52x512

        P3 = self.up3(P3)
        P3 = self.conv3(P3)

        # P3 = self.up4(P3)
        # P3 = self.conv4(P3)

        out = self.oup(P3)
        #print(f"out.shape is {out.shape}")

        return F.sigmoid(out)
