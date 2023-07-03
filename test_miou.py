import argparse
import cv2
import time
import datetime
import os
import random
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from model import my_cnn, stage2_cnn, ResNet, Wide_Deep, Fibo_Dense, Fibo_2, small, Unet, Densenet, Dense_Fibo
from DeepLab import DeepLabV3Plus
from PSPnet import PSPNet
from thop import profile

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

def mIOU(outputs, targets):
    #outputs=np.tanh(outputs)/2+0.5
    # num_pic = outputs.shape[0]
    # outputs = torch.abs(outputs).view(num_pic, -1).detach().cpu().numpy()
    # targets = targets.view(num_pic, -1).detach().cpu().numpy()
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum() + 1e-7
    iou = intersection / (union - intersection)
    return iou.sum()

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda:0'
    model = torch.load(args.savedir).to(args.device)#Fibo_2(input_channel=3, mid_channel=12, labels=1)#Dense_Fibo(input_channel=3, mid_channel=12, labels=1)#Unet()#DeepLabV3Plus(num_classes=3)#PSPNet(num_classes=3, downsample_factor=8)#
    #my_cnn(input_channel=3, mid_channel=12, layers=12, labels=3)#Wide_Deep(input_channel=3, mid_channel=24, labels=3)#small(input_channel=3, mid_channel=12, layers=1, labels=1)#
    #stage2_cnn(input_channel=3, mid_channel=16, labels=3)#ResNet()#Densenet(input_channel=3, mid_channel=12, labels=1)#Fibo_Dense(input_channel=3, mid_channel=12, labels=1)#
    model.eval()
        #算力计算
    # input = torch.randn(1, 3, 640, 360).to('cuda') #模型输入的形状,batch_size=1
    # flops, params = profile(model, inputs=(input, ))
    # print("Flops(G):",flops/1e9, "Parameters(M):",params/1e6) #flops单位G，para单位M
    iou_s,iou_m,iou_l=0,0,0
    num_s,num_m,num_l=0,0,0
    with open("D:/microlight/test.txt", "r") as f:
        for line in f.readlines():
            temp_image = cv2.imread('D:/microlight/image/' + line.strip('\n')).astype(np.float16)/255.0
            img_t = torch.tensor(np.transpose(temp_image, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
            result = model(img_t).squeeze().detach().cpu().numpy()
            temp_label = cv2.imread('D:/microlight/label/' + line.strip('\n')).astype(np.float16)/255.0
            # plt.imshow(np.transpose(np.tanh(result)/2+0.5,(1,2,0)))
            # plt.show()
            # print(result.max(),result.min())
            if(temp_label[0].sum()<64):
                iou_s+=mIOU(result[0,:,:],temp_label[:,:,0])
                num_s+=1
            elif(temp_label[0].sum()<256):
                iou_m+=mIOU(result[0,:,:],temp_label[:,:,0])
                num_m+=1
            else:
                iou_l+=mIOU(result[0,:,:],temp_label[:,:,0])
                num_l+=1
    print("iou_s:{:.5f}  num: {:4d} || iou_m:{:.5f}  num: {:4d}  || iou_l:{:.5f}  num: {:4d} ".format(iou_s/num_s, num_s, iou_m/num_m, num_m, iou_l/num_l, int(num_l)))
    print("iou_test:{:.5f}".format((iou_s+iou_m+iou_l)/(num_s+num_m+num_l)))
    file_list = os.listdir('D:/Download/')
    for i in range(3):
        img = cv2.imread('D:/Download/' + str(i)+'.png').astype(np.int32)
        img_t = torch.tensor(np.transpose(img.astype(np.float16)/255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
        result = model(img_t).squeeze().detach().cpu().numpy()
        img[:,:,0]+= (100*result[0,:,:]).astype(np.int32)
        img[:,:,1]-= (50*result[0,:,:]).astype(np.int32)
        img[:,:,2]-= (50*result[0,:,:]).astype(np.int32)
        img[img>255]=255
        img[img<0]=0
        cv2.imwrite('D:/Download/s1' + str(i)+'.png', img.astype(np.uint8))