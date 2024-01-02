import argparse
import cv2
import time
import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from model import my_cnn, stage2_cnn, ResNet, Wide_Deep, Fibo_Dense, Fibo_2, small, Unet, Densenet, Dense_Fibo
from DeepLab import DeepLabV3Plus
from PSPnet import PSPNet
from shuffle_v2 import shufflenet_1x,shufflenet_1x_se_res,shuffle_Unet
from mobilenet import MobileNet,MobileNetV2
from Unet_mobile import Mobile_UNet

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--workers', '-j', type=int, default=0,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--batchsize', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 0.02)')
    # cuda setting
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device of training')
    # checkpoint and log
    parser.add_argument('--savedir', default='D:/microlight/models/shufflev2_207.pth',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()
    return args


class dataset(data.Dataset):
    def __init__(self, filedir):
        self.images = []
        self.target = []
        with open("D:/microlight/"+filedir, "r") as f:
            for line in f.readlines():
                temp_image = cv2.imread('D:/microlight/image/' + line.strip('\n')).astype(np.float16)/255.0
                temp_label = cv2.imread('D:/microlight/label/' + line.strip('\n')).astype(np.float16)/255.0
                temp_label = np.transpose(temp_label, (2, 0, 1))
                # kernel = np.ones((3, 3), np.uint8)
                # temp_label[0, :, :] = cv2.dilate(temp_label[0, :, :], kernel)-cv2.erode(temp_label[0, :, :], kernel)
                # new_temp = temp_label.sum(0)
                # temp_label[1, new_temp<1] = 1.0
                self.images.append(np.transpose(temp_image, (2, 0, 1)))
                self.target.append(temp_label)


    def __getitem__(self, index):
        img = self.images[index]
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()*(7**self.gamma)
class DiceLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predictive, target):
        intersection =  torch.sum(predictive * target)
        union = torch.sum(predictive) + torch.sum(target) + self.eps#
        loss = 1 - intersection / union
        return loss
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.device = torch.device(args.device)

        # dataset and dataloader
        train_dataset = dataset('train.txt')
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batchsize, drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)
        test_dataset = dataset('test.txt')
        self.test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batchsize, drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)
        self.train_num = train_dataset.__len__()
        self.test_num = test_dataset.__len__()
        print("Train images:", self.train_num, "Test images:", self.test_num)
        # create network
        self.model = model.to(self.device)

        # create criterion
        self.criterion = FocalLoss()#DiceLoss()#nn.MSELoss()#


    def train(self):
        start_time = time.time()
        max_IOU = 0.04
        for epoch in range(70, args.epochs):
            self.model.train()
            optimizer =  torch.optim.AdamW(model.parameters(), lr=self.args.lr/(epoch+10), weight_decay=5e-4)#torch.optim.SGD(model.parameters(), lr=self.args.lr/5, weight_decay=5e-4)#
            
            if (epoch>29):
                self.criterion = DiceLoss()
            epoch_loss, miou, test_miou = 0.0, 0.0, 0.0 
            for iteration, (images, targets) in enumerate(self.train_loader):
                iteration = iteration + 1

                images =  torch.as_tensor(images, dtype=torch.float32).to(self.device)
                targets = torch.as_tensor(targets, dtype=torch.float32).to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)#.unsqueeze(1)
                epoch_loss += loss
                miou += mIOU(outputs, targets)#.unsqueeze(1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for _, (images, targets) in enumerate(self.test_loader):
                    images =  torch.as_tensor(images, dtype=torch.float32).to(self.device)
                    targets = torch.as_tensor(targets, dtype=torch.float32).to(self.device)
                    outputs = self.model(images)
                    test_miou += mIOU(outputs, targets.unsqueeze(1))

            print("epoch:{:2d}  Lr: {:.5f} || Loss: {:.5f} || Cost Time: {} || mIOU: {:.5f} || Test_mIOU: {:.5f}".format(
                epoch+1, optimizer.param_groups[0]['lr'], epoch_loss, str(datetime.timedelta(seconds=int(time.time() - start_time))), miou/self.train_num, test_miou/self.test_num))
            if (max_IOU < test_miou/self.test_num):
                max_IOU = test_miou/self.test_num
                torch.save(self.model, self.args.savedir)
        return max_IOU


def mIOU(outputs, targets):
    num_pic = outputs.shape[0]
    outputs = torch.abs(outputs).view(num_pic, -1).detach().cpu().numpy()
    targets = targets.view(num_pic, -1).detach().cpu().numpy()
    intersection = (outputs * targets).sum(1)
    union = outputs.sum(1) + targets.sum(1) + 1e-7
    iou = intersection / (union - intersection)
    return iou.sum()

if __name__ == '__main__':
    args = parse_args()
    model = torch.load(args.savedir).to(args.device)#Mobile_UNet(n_channels=3,num_classes=3)#shufflenet_1x_se_res(num_classes=3)#PSPNet(num_classes=3, downsample_factor=8)#Fibo_2(input_channel=3, mid_channel=12, labels=1)#Dense_Fibo(input_channel=3, mid_channel=12, labels=1)#Unet()#DeepLabV3Plus(num_classes=3)#
    #my_cnn(input_channel=3, mid_channel=12, layers=12, labels=3)#Wide_Deep(input_channel=3, mid_channel=24, labels=3)#small(input_channel=3, mid_channel=12, layers=1, labels=1)#
    #stage2_cnn(input_channel=3, mid_channel=16, labels=3)#ResNet()#Densenet(input_channel=3, mid_channel=12, labels=1)#Fibo_Dense(input_channel=3, mid_channel=12, labels=1)#
    trainer = Trainer(args, model)
    iou = trainer.train()
    print("best perform IoU:", iou)
