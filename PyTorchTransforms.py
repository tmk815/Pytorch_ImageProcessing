import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('content/owl.jpg')
plt.imshow(image)  # 描画

my_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

img = my_transform(image)
print(img.size())  # [c,h,w]


def imshow(tensor):
    image = tensor.clone().detach()  # copy
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = unloader(image)
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


imshow(img)

x = img.unsqueeze_(0)
print(x.size())  # [mb,c,h,w]

# 平滑化
k = 9
kernel = torch.FloatTensor([[1 / k, 1 / k, 1 / k],
                            [1 / k, 1 / k, 1 / k],
                            [1 / k, 1 / k, 1 / k]])

blur_filter = kernel.expand(1, 1, 3, 3)  # [h,w] ⇒ [mb,c,h,w]
blur_img = F.conv2d(x, blur_filter)

imshow(blur_img)  # 描画

# Sobelフィルタ
kernel = torch.FloatTensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

sobel_filter = kernel.expand(1, 1, 3, 3)
sobel_img = F.conv2d(x, sobel_filter)

imshow(sobel_img)  # 描画

# 鮮鋭化
k = 0.3  # Shape Strength
kernel = torch.FloatTensor([[0, -k, 0],
                            [-k, 1 + 4 * k, -k],
                            [0, -k, 0]])

kernel2 = torch.FloatTensor([[-k, -k, -k],
                             [-k, 1 + (8 * k), -k],
                             [-k, -k, -k]])

shape_filter = kernel.expand(1, 1, 3, 3)
shape_filter2 = kernel2.expand(1, 1, 3, 3)
shape_img = F.conv2d(x, shape_filter)
shape_img2 = F.conv2d(x, shape_filter2)

imshow(shape_img)  # 描画
imshow(shape_img2)  # 描画

# PyTorchのCNNの重みとバイアスを確認
conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)

print(conv1.state_dict().keys())
print(conv1.state_dict()['weight'])
print(conv1.state_dict()['bias'])

# PyTorchでSobelレイヤー作成
sobel_kernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
sobel_kernel.state_dict()['weight'][0] = torch.FloatTensor([[1, 0, -1],
                                                            [2, 0, -2],
                                                            [1, 0, -1]])
sobel_kernel.state_dict()['bias'].zero_()

print(sobel_kernel.state_dict().keys())
print(sobel_kernel.state_dict()['weight'])
print(sobel_kernel.state_dict()['bias'])
