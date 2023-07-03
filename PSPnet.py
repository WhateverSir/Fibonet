#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

#from base.base_model import BaseModel
from backbonds import get_resnet#, VGG


#------------------------------------------------------------------------------
#  PSP
#------------------------------------------------------------------------------
class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) 
                                                        for pool_size in pool_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(pool_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output



#------------------------------------------------------------------------------
#  PSPNet
#------------------------------------------------------------------------------

class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="resnet18", pretrained=False, aux_branch=False):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone=='resnet18':
            num_layers = 18
            aux_channel = 64
            out_channel = 512
            self.backbone = get_resnet(num_layers=num_layers, num_classes=None)


        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel//8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1)
            )
		
        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self._run_backbone_resnet(x)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return F.sigmoid(output)

    def _run_backbone_resnet(self, input):
		# Stage1
        x1 = self.backbone.conv1(input)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
		# Stage2
        x2 = self.backbone.maxpool(x1)
        x2 = self.backbone.layer1(x2)
		# Stage3
        x3 = self.backbone.layer2(x2)
		# Stage4
        x4 = self.backbone.layer3(x3)
		# Stage5
        x5 = self.backbone.layer4(x4)
		# Output
        return x2, x5

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()