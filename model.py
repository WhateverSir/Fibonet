import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#二值化层定义
class BinarizedF(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        a = torch.ones_like(input)
        b = -torch.ones_like(input)
        output = torch.where(input>=0,a,b)
        return output
    def backward(self, output_grad):
        input, = self.saved_tensors
        input_abs = torch.abs(input)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        input_grad = torch.where(input_abs<=1,ones, zeros)
        return input_grad

class my_cnn(nn.Module):
    def __init__(self, input_channel, mid_channel, layers, labels, out=True, dropout=0.2):
        super(my_cnn, self).__init__()
        self.out = out
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.Conv2d(in_channels=input_channel, out_channels=mid_channel, kernel_size=7, stride=1, padding=3),
            
            nn.ReLU()
        )
        conv_list = []
        for i in range(layers):
            conv_list.append(nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=1, padding=1))
            #conv_list.append(nn.Dropout(dropout))
            conv_list.append(nn.BatchNorm2d(mid_channel))
            conv_list.append(nn.ReLU())
        self.conv2 = nn.Sequential(*conv_list)
        self.out_layer = nn.Sequential(nn.Conv2d(in_channels=mid_channel, out_channels=labels, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if(self.out):
            x = torch.sigmoid(self.out_layer(x))
        return x

class stage2_cnn(nn.Module):
    def __init__(self, input_channel, mid_channel, labels):
        super(stage2_cnn, self).__init__()
        self.cnn1 = my_cnn(input_channel=input_channel, mid_channel=mid_channel, layers=6, labels=labels, out=False)
        #self.cnn1.requires_grad = False
        # conv_list = []
        # for i in range(4):
        #     conv_list.append(nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=1, padding=1))
        #     conv_list.append(nn.BatchNorm2d(mid_channel))
        #     conv_list.append(nn.ReLU())
        # conv_list.append(nn.Conv2d(in_channels=mid_channel, out_channels=labels, kernel_size=1, stride=1, padding=0))
        # self.cnn2 = nn.Sequential(*conv_list)
        self.cnn2 = my_cnn(input_channel=mid_channel, mid_channel=mid_channel, layers=4, labels=labels)
        self.mid_out = nn.Conv2d(in_channels=mid_channel, out_channels=labels, kernel_size=1, stride=1, padding=0)
        #self.mid_out.requires_grad = False
        self.transition_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.out_layer = nn.Conv2d(in_channels=labels*2, out_channels=labels, kernel_size=1, stride=1, padding=0)
        self.out_layer.bias.data.fill_(-4.6)
    def forward(self, x):
        x = self.cnn1(x)
        midx = self.mid_out(x)
        x = self.transition_layer(x)
        x = self.cnn2(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([midx, x], 1)
        x = self.out_layer(x)
        return torch.sigmoid(midx)

        
#resnet 实现
class BasicBlock(nn.Module):
    expansion = 1
	#inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
		#把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], labels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1])
        self.mid_out = nn.Conv2d(in_channels=64, out_channels=labels, kernel_size=1, stride=1, padding=0)
        self.layer3 = self._make_layer(block, 128, layers[2])
        self.layer4 = self._make_layer(block, 256, layers[3])
        self.avgpool = nn.Conv2d(in_channels=256, out_channels=labels, kernel_size=1, stride=1, padding=0)
        self.out_layer = nn.Conv2d(in_channels=labels*2, out_channels=labels, kernel_size=1, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.out_layer.bias.data.fill_(-4.6)
    def _make_layer(self, block, planes, blocks, stride=1):
        #downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        #self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        #在shotcut中若维度或者feature_size不一致则需要downsample 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        #只在这里传递了stride=2的参数，因而一个box_block中的图片大小只在第一次除以2
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        midx = F.interpolate(self.mid_out(x), scale_factor=2)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.interpolate(x, scale_factor=4)

        x = self.avgpool(x)
        x = torch.cat([midx, x], 1)
        x = self.out_layer(x)
        return torch.sigmoid(x)#softmax(x, dim=1)


class Wide_Deep(nn.Module):
    def __init__(self, input_channel, mid_channel, labels):
        super(Wide_Deep, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.mid_out = nn.Conv2d(in_channels=64, out_channels=labels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dnn = my_cnn(input_channel=64, mid_channel=mid_channel, layers=10, labels=labels, out=True)
        self.out_layer = nn.Conv2d(in_channels=labels*2, out_channels=labels, kernel_size=1, stride=1, padding=0)
        self.out_layer.bias.data.fill_(-4.6)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        midx = self.mid_out(x)
        x = self.maxpool(x)
        x = self.dnn(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([midx, x], 1)
        x = self.out_layer(x)
        return torch.sigmoid(x)

class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Dense_block(nn.Module):
    def __init__(self, in_channel, grow_channel, layer_num, labels=None):
        super(Dense_block, self).__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(conv_block(in_channel=in_channel+grow_channel*i, out_channel= grow_channel))
        if (labels is not None):
            self.out_layer = nn.Conv2d(in_channel+grow_channel*layer_num, labels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        for i in range(self.layer_num):
            x1 = self.layers[i](x)
            x = torch.cat([x1, x], 1)
        x = self.out_layer(x)
        return x

class Fibo_block(nn.Module):
    def __init__(self, in_channel, layer_num, grow_rate=0.618, labels=None):
        super(Fibo_block, self).__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=5, stride=1, padding=2, bias=False)
        channel1, channel2 = in_channel, in_channel
        for i in range(layer_num):
            temp = math.floor((channel1 + channel2 )* grow_rate)
            self.layers.append(conv_block(in_channel=channel1+channel2, out_channel= temp))
            channel1 = channel2
            channel2 = temp 
        if (labels is not None):
            self.out_layer = nn.Conv2d(channel2, labels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x1 = x
        x2 = self.conv1(x)
        for i in range(self.layer_num):
            x = torch.cat([x1, x2], 1)
            x = self.layers[i](x) 
            x1 = x2
            x2 = x   
        x = self.out_layer(x2)
        return x2, x

class Fibo_Dense(nn.Module):
    def __init__(self, input_channel, mid_channel, labels):
        super(Fibo_Dense, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv1 = BasicBlock(inplanes=mid_channel, planes=mid_channel)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.Fibo = Fibo_block(in_channel=mid_channel, layer_num=6, labels=labels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Dense = Dense_block(in_channel=27, grow_channel=mid_channel, layer_num=8, labels=labels)
        self.out_layer = nn.Conv2d(labels*2, labels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x1 = self.conv1(self.avgpool(x))
        x = self.conv1(x) +F.interpolate(x1, scale_factor=2)       
        x = self.avgpool(self.res_conv1(x))
        x, midx = self.Fibo(x)
        x = self.maxpool(x)
        x = self.Dense(x)
        x = torch.cat([midx, F.interpolate(x, scale_factor=2)], 1)
        x = self.out_layer(F.interpolate(x, scale_factor=2))
        return torch.sigmoid(x)
class Dense_Fibo(nn.Module):
    def __init__(self, input_channel, mid_channel, labels):
        super(Dense_Fibo, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv1 = BasicBlock(inplanes=mid_channel, planes=mid_channel)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.Fibo = Fibo_block(in_channel=labels, layer_num=8, labels=labels, grow_rate=1.)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Dense = Dense_block(in_channel=mid_channel, grow_channel=mid_channel, layer_num=6, labels=labels)
        self.out_layer = nn.Conv2d(labels*2, labels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x1 = self.conv1(self.avgpool(x))
        x = self.conv1(x) +F.interpolate(x1, scale_factor=2)       
        x = self.avgpool(self.res_conv1(x))
        midx = self.Dense(x)
        x = self.maxpool(midx)
        _, x = self.Fibo(x)
        x = torch.cat([midx, F.interpolate(x, scale_factor=2)], 1)
        x = self.out_layer(F.interpolate(x, scale_factor=2))
        return torch.sigmoid(x)

class Densenet(nn.Module):
    def __init__(self, input_channel, mid_channel, labels):
        super(Densenet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.dense = Dense_block(in_channel=mid_channel, grow_channel=mid_channel, layer_num=6, labels=labels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Dense = Dense_block(in_channel=labels, grow_channel=mid_channel, layer_num=8, labels=labels)
        self.out_layer = nn.Conv2d(labels*2, labels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x1 = self.conv1(x)
        midx = self.dense(self.maxpool(x1))
        x = self.maxpool(midx)
        x = self.Dense(x)
        x = torch.cat([midx, F.interpolate(x, scale_factor=2)], 1)
        x = self.out_layer(F.interpolate(x, scale_factor=2))
        return torch.sigmoid(x)        
class Fibo_2(nn.Module):
    def __init__(self, input_channel, mid_channel, labels):
        super(Fibo_2, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.Fibo = Fibo_block(in_channel=mid_channel, layer_num=6, labels=labels, grow_rate=0.75)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Fibo2 = Fibo_block(in_channel=67, layer_num=8, labels=labels, grow_rate=0.382)
        self.out_layer = nn.Conv2d(labels*2, labels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        x, midx = self.Fibo(x)
        x = self.maxpool(x)
        _temp, x = self.Fibo2(x)
        x = torch.cat([midx, F.interpolate(x, scale_factor=2)], 1)
        x = self.out_layer(x)
        return torch.sigmoid(x)

class small(nn.Module):
    def __init__(self, input_channel, mid_channel, layers, labels):
        super(small, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, mid_channel, kernel_size=25, stride=1, padding=12, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        conv_list = []
        for i in range(layers):
            conv_list.append(BasicBlock(inplanes=mid_channel, planes=mid_channel, stride=1))
        self.conv2 = nn.Sequential(*conv_list)
        self.out_layer = nn.Conv2d(mid_channel, labels, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        x = self.conv1(input)
        x1 = self.avgpool(x)
        x = self.conv2(x)
        x1 = self.conv2(x1)
        x = self.out_layer(x+F.interpolate(x1, scale_factor=2))
        return torch.sigmoid(x)

class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.layer1_conv = double_conv2d_bn(3,8)
        self.layer2_conv = double_conv2d_bn(8,16)
        self.layer3_conv = double_conv2d_bn(16,32)
        self.layer4_conv = double_conv2d_bn(32,64)
        self.layer5_conv = double_conv2d_bn(64,128)
        self.layer6_conv = double_conv2d_bn(128,64)
        self.layer7_conv = double_conv2d_bn(64,32)
        self.layer8_conv = double_conv2d_bn(32,16)
        self.layer9_conv = double_conv2d_bn(16,8)
        self.layer10_conv = nn.Conv2d(8,3,kernel_size=3, stride=1,padding=1,bias=True)
        
        self.deconv1 = deconv2d_bn(128,64)
        self.deconv2 = deconv2d_bn(64,32)
        self.deconv3 = deconv2d_bn(32,16)
        self.deconv4 = deconv2d_bn(16,8)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1,2)
        
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2,2)
        
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3,2)
        
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4,2)
        
        #conv5 = self.layer5_conv(pool4)
        
        # convt1 = self.deconv1(conv5)
        # concat1 = torch.cat([convt1,conv4],dim=1)
        # conv6 = self.layer6_conv(concat1)
        
        convt2 = self.deconv2(conv4)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.layer7_conv(concat2)
        
        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.layer8_conv(concat3)
        
        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,conv1],dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)
        return outp
