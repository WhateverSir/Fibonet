from matplotlib.pyplot import xcorr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#shuffle层定义
def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    # bs, chnls, h, w = x.data.size()
    # if chnls % groups:
    #     return x
    # chnls_per_group = chnls // groups
    # x = x.view(bs, groups, chnls_per_group, h, w)
    # #x = torch.transpose(x, 1, 2).contiguous()
    # x = x.permute(0,2,1,3,4).contiguous()
    # x = x.view(bs, chnls, h, w)
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    return out

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class DSampling(nn.Module):
    """Spatial down sampling of SuffleNet-v2"""

    def __init__(self, in_chnls, groups=2):
        super(DSampling, self).__init__()
        self.groups = groups
        self.dwconv_l1 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1,  # down-sampling, depth-wise conv.
                                   groups=in_chnls, activation=None)
        self.conv_l2 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.conv_r1 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.dwconv_r2 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1, groups=in_chnls, activation=False)
        self.conv_r3 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)

    def forward(self, x):
        # left path
        out_l = self.dwconv_l1(x)
        out_l = self.conv_l2(out_l)

        # right path
        out_r = self.conv_r1(x)
        out_r = self.dwconv_r2(out_r)
        out_r = self.conv_r3(out_r)

        # concatenate
        out = torch.cat((out_l, out_r), 1)
        return shuffle_chnls(out, self.groups)


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)

class BasicUnit(nn.Module):
    """Basic Unit of ShuffleNet-v2"""

    def __init__(self, in_chnls, out_chnls, is_se=False, is_residual=False, c_ratio=0.5, groups=2):
        super(BasicUnit, self).__init__()
        self.is_se, self.is_res = is_se, is_residual
        self.l_chnls = int(in_chnls * c_ratio)
        self.r_chnls = in_chnls - self.l_chnls
        self.ro_chnls = out_chnls - self.l_chnls
        self.groups = groups

        # layers
        self.conv1 = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0)
        self.dwconv2 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 3, 1, 1,  # same padding, depthwise conv
                                 groups=self.ro_chnls, activation=None)
        act = False if self.is_res else True
        self.conv3 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 1, 1, 0, activation=act)
        if self.is_se:
            self.se = SE(self.ro_chnls, 16)
        if self.is_res:
            self.shortcut = nn.Sequential()
            if self.r_chnls != self.ro_chnls:
                self.shortcut = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0, activation=False)

    def forward(self, x):
        # x_l = x[:, :self.l_chnls, :, :]
        # x_r = x[:, self.l_chnls:, :, :]
        x_l, x_r = torch.split(x,self.l_chnls,dim=1)

        # right path
        out_r = self.conv1(x_r)
        out_r = self.dwconv2(out_r)
        out_r = self.conv3(out_r)
        if self.is_se:
            coefficient = self.se(out_r)
            out_r *= coefficient
        if self.is_res:
            out_r += self.shortcut(x_r)

        # concatenate
        out = torch.cat((x_l, out_r), 1)
        return shuffle_chnls(out, self.groups)
class ShrinkUnit(nn.Module):
    """Basic Unit of ShuffleNet-v2"""

    def __init__(self, in_chnls, out_chnls, is_se=False, is_residual=False, c_ratio=0.5, groups=2):
        super(ShrinkUnit, self).__init__()
        self.is_se, self.is_res = is_se, is_residual
        self.l_chnls = int(in_chnls * c_ratio)
        self.r_chnls = in_chnls - self.l_chnls
        self.lo_chnls = int(out_chnls * c_ratio)
        self.ro_chnls = out_chnls - self.lo_chnls
        self.groups = groups

        # layers
        # self.dwconv_l1 = BN_Conv2d(self.l_chnls, self.l_chnls, 3, 1, 1,  # same padding, depth-wise conv.
        #                            groups=in_chnls, activation=None)
        self.conv_l2 = BN_Conv2d(self.l_chnls, self.lo_chnls, 1, 1, 0)
        self.conv1 = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0)
        self.dwconv2 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 3, 1, 1,  # same padding, depthwise conv
                                 groups=self.ro_chnls, activation=None)
        act = False if self.is_res else True
        self.conv3 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 1, 1, 0, activation=act)
        if self.is_se:
            self.se = SE(self.ro_chnls, 16)
        if self.is_res:
            self.shortcut = nn.Sequential()
            if self.r_chnls != self.ro_chnls:
                self.shortcut = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0, activation=False)

    def forward(self, x):
        # x_l = x[:, :self.l_chnls, :, :]
        # x_r = x[:, self.l_chnls:, :, :]
        x_l, x_r = torch.split(x,self.l_chnls,dim=1)
        # out_l = self.dwconv_l1(x_l)
        out_l = self.conv_l2(x_l)
        # right path
        out_r = self.conv1(x_r)
        out_r = self.dwconv2(out_r)
        out_r = self.conv3(out_r)
        if self.is_se:
            coefficient = self.se(out_r)
            out_r *= coefficient
        if self.is_res:
            out_r += self.shortcut(x_r)

        # concatenate
        out = torch.cat((out_l, out_r), 1)
        return shuffle_chnls(out, self.groups)
class ShuffleNet_v2(nn.Module):
    """ShuffleNet-v2"""

    _defaults = {
        "sets": {0.5, 1, 1.5, 2},
        "units": [3, 7, 3],
        "chnl_sets": {0.5: [24, 48, 96, 192, 1024],
                      1: [24, 116, 232, 464, 1024],
                      1.5: [24, 176, 352, 704, 1024],
                      2: [24, 244, 488, 976, 2048]}
    }

    def __init__(self, scale, num_cls, is_se=False, is_res=False) -> object:
        super(ShuffleNet_v2, self).__init__()
        self.__dict__.update(self._defaults)
        assert (scale in self.sets)
        self.is_se = is_se
        self.is_res = is_res
        self.chnls = self.chnl_sets[scale]

        # make layers
        self.conv1 = BN_Conv2d(3, self.chnls[0], 3, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.stage2 = self.__make_stage(self.chnls[0], self.chnls[1], self.units[0])
        self.stage3 = self.__make_stage(self.chnls[1], self.chnls[2], self.units[1])
        self.stage4 = self.__make_stage(self.chnls[2], self.chnls[3], self.units[2])
        self.conv5 = BN_Conv2d(self.chnls[3], self.chnls[1], 1, 1, 0)
        #self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.globalpool = nn.Conv2d(self.chnls[1], num_cls, 3, 1, 1,bias=True)
        self.body = self.__make_body()
        self.fc = nn.Conv2d(num_cls, num_cls, 3, 1, 1,bias=True)#nn.Linear(self.chnls[4], num_cls)

    def __make_stage(self, in_chnls, out_chnls, units):
        layers = [DSampling(in_chnls),
                  BasicUnit(2 * in_chnls, out_chnls, self.is_se, self.is_res)]
        for _ in range(units-1):
            layers.append(BasicUnit(out_chnls, out_chnls, self.is_se, self.is_res))
        return nn.Sequential(*layers)

    def __make_body(self):
        return nn.Sequential(
            self.conv1,   self.stage2,  self.stage3,
            self.stage4, self.conv5, self.globalpool
        )

    def forward(self, x):
        out = self.body(x)
        out = F.interpolate(out, scale_factor=8)
        # out.view(out.size(0), out.size(1))
        out = self.fc(out)
        return F.sigmoid(out)


"""
API
"""


def shufflenet_0_5x(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes)


def shufflenet_0_5x_se(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_se=True)


def shufflenet_0_5x_res(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_res=True)


def shufflenet_0_5x_se_res(num_classes=1000):
    return ShuffleNet_v2(0.5, num_classes, is_se=True, is_res=True)


def shufflenet_1x(num_classes=1000):
    return ShuffleNet_v2(1, num_classes)


def shufflenet_1x_se(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_se=True)


def shufflenet_1x_res(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_res=True)


def shufflenet_1x_se_res(num_classes=1000):
    return ShuffleNet_v2(1, num_classes, is_se=True, is_res=True)


def shufflenet_1_5x(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes)


def shufflenet_1_5x_se(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_se=True)


def shufflenet_1_5x_res(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_res=True)


def shufflenet_1_5x_se_res(num_classes=1000):
    return ShuffleNet_v2(1.5, num_classes, is_se=True, is_res=True)


def shufflenet_2x(num_classes=1000):
    return ShuffleNet_v2(2, num_classes)


def shufflenet_2x_se(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_se=True)


def shufflenet_2x_res(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_res=True)


def shufflenet_2x_se_res(num_classes=1000):
    return ShuffleNet_v2(2, num_classes, is_se=True, is_res=True)

class double_conv2d_bn_sf(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1,groups=2):
        super(double_conv2d_bn_sf,self).__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=False)
        self.conv2 = nn.Conv2d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,groups=out_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return shuffle_chnls(out, self.groups)
    
class deconv2d_bn_sf(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn_sf,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
class shuffle_Unet(nn.Module):
    def __init__(self):
        super(shuffle_Unet,self).__init__()
        self.layer1_conv = double_conv2d_bn_sf(3,8)
        self.layer2_conv = self.__make_stage(8,16)
        self.layer3_conv = self.__make_stage(16,32)
        self.layer4_conv = self.__make_stage(32,64)
        self.layer5_conv = self.__make_stage(64,128)
        self.layer6_conv = self.__make_stage(128,64, shrink=True)
        self.layer7_conv = self.__make_stage(64,32, shrink=True)
        self.layer8_conv = self.__make_stage(32,16, shrink=True)
        self.layer9_conv = self.__make_stage(16,8, shrink=True)
        self.layer10_conv = nn.Conv2d(8,3,kernel_size=3,stride=1,padding=1,bias=True)
        
        self.deconv1 = deconv2d_bn_sf(128,64,kernel_size=5, strides=5)
        self.deconv2 = deconv2d_bn_sf(64,32)
        self.deconv3 = deconv2d_bn_sf(32,16)
        self.deconv4 = deconv2d_bn_sf(16,8)
        self.header_heat = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.Conv2d(16, 3, kernel_size=1))
        self.out = nn.Conv2d(3, 3, kernel_size=5, padding=2) 
    def __make_stage(self, in_chnls, out_chnls, units=2, shrink=False):
        if(shrink):
            layers = [ShrinkUnit(in_chnls, out_chnls)]
        else:
            layers = [BasicUnit(in_chnls, out_chnls)]
        for _ in range(units-1):
            layers.append(BasicUnit(out_chnls, out_chnls))
        return nn.Sequential(*layers)    
    def forward(self,x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1,2)
        
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2,2)
        
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3,2)
        
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4,5)
        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1,conv4],dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.layer7_conv(concat2)
        
        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.layer8_conv(concat3)
        
        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,conv1],dim=1)
        conv9 = self.layer9_conv(concat4)
        out_heat = self.header_heat(conv9)
        out = self.out(out_heat)
        return F.sigmoid(out)

if __name__ == '__main__':
    model = shufflenet_0_5x()
    torch.save(model, 'D:/Download/shufflev2.pth')