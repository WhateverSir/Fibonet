# Fibonet: 一个用于图像分割的轻量且高效的神经网络
## 介绍

Fibonet是一个轻量且高效的神经网络，用于图像分割。该网络采用了一种名为Fibonacci卷积的新型卷积操作，能够在保持高准确性的同时，大大减少网络参数和计算量。
## 环境要求
- Python 3.8以上版本
- PyTorch 1.9.1及以上版本
- CUDA 11.4及以上版本（如果您想使用GPU进行加速）
- NumPy库

请确保您的计算机满足以上要求，并安装了所需的依赖项，以便成功运行Fibonet。
## 代码

- `DeepLab.py`：DeepLab是一个基于深度学习的语义分割模型，该文件包含了DeepLab模型的实现代码。
- `PSPnet.py`：PSPnet是一种具有多尺度池化的语义分割模型，该文件包含了PSPnet模型的实现代码。
- `README.md`：本文件，提供了项目的介绍和使用说明。
- `Unet_mobile.py`：Unet_mobile是一种轻量级的语义分割模型，该文件包含了Unet_mobile模型的实现代码。
- `backbonds.py`：该文件包含了Fibonacci卷积操作的实现代码。
- `mobilenet.py`：MobileNet是一种轻量级的卷积神经网络，该文件包含了MobileNet模型的实现代码。
- `model.py`：该文件包含了Fibonet模型的实现代码。
- `shuffle_v2.py`：ShuffleNet V2是一种具有低计算复杂度和高准确性的卷积神经网络，该文件包含了骨干网络ShuffleNet/Shuffle-UNet的实现代码。
- `test_miou.py`：该文件包含了计算图像分割平均IoU（Intersection over Union）的代码。
- `train.py`：该文件包含了训练Fibonet模型所需的代码。

## 使用

要使用Fibonet进行图像分割，您需要执行以下步骤：

1. 下载并安装PyTorch。
2. 下载本项目中的所有代码文件。
3. 使用train.py文件训练Fibonet模型。
4. 使用test_miou.py文件测试训练好的模型在图像分割上的性能。

请注意，训练Fibonet模型可能需要较长时间，具体时间取决于您使用的数据集和计算机性能。

## 引用

如果您在研究中使用了Fibonet，请引用以下论文：
Wu Ruohao, et al. Fibonet: A Light-weight and Efficient Neural Network for Image Segmentation[C]//2023 IEEE International Conference on Image Processing (ICIP). IEEE, Kuala Lumpur, Malaysia, October 8-11, 2023: 1345-1349. 
