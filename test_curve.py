import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision.models import resnet18
from tqdm import tqdm
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'Perpetua'

fontsize = 11
plt.figure(figsize=(12,4))

data1 = 'unet/net40.txt'
with open(data1, "r",encoding='utf-8') as f:  # 打开文件
    s = f.read()  # 读取文件
    # print(s)
acc1 = []
for i in re.findall("test_accuracy:-*[0-9].[0-9]+", s):
    acc1.append(float(i.replace('test_accuracy:','')))
print(acc1)

data2 = 'unet/net41.txt'
with open(data2, "r",encoding='utf-8') as f:  # 打开文件
    s = f.read()  # 读取文件
    # print(s)
acc2 = []
for i in re.findall("test_accuracy:-*[0-9].[0-9]+", s):
    acc2.append(float(i.replace('test_accuracy:','')))
print(acc2)

x = [j for j in range(1,len(acc1)+1)]

plt.subplot(1,3,1)
plt.plot(x, acc1, '^g--', label="1000 training data set without Constraint", markersize=3)
plt.plot(x, acc2, 'ob-', label="1000 training data set with Constraint", markersize=3)
plt.legend(fontsize=fontsize,loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('SSIM')

ymin = 0
ymax = 1
axes = plt.gca()
axes.set_ylim([ymin,ymax])

data1 = 'unet/net34.txt'
with open(data1, "r",encoding='utf-8') as f:  # 打开文件
    s = f.read()  # 读取文件
    # print(s)
acc1 = []
for i in re.findall("test_accuracy:-*[0-9].[0-9]+", s):
    acc1.append(float(i.replace('test_accuracy:','')))
print(acc1)

data2 = 'unet/net35.txt'
with open(data2, "r",encoding='utf-8') as f:  # 打开文件
    s = f.read()  # 读取文件
    # print(s)
acc2 = []
for i in re.findall("test_accuracy:-*[0-9].[0-9]+", s):
    acc2.append(float(i.replace('test_accuracy:','')))
print(acc2)

x = [j for j in range(1,len(acc1)+1)]

plt.subplot(1,3,2)
plt.plot(x, acc1, '^g--', label="5000 training data set without Constraint", markersize=3)
plt.plot(x, acc2, 'ob-', label="5000 training data set with Constraint", markersize=3)
plt.legend(fontsize=fontsize,loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('SSIM')

ymin = 0
ymax = 1
axes = plt.gca()
axes.set_ylim([ymin,ymax])

data1 = 'unet/net43.txt'
with open(data1, "r",encoding='utf-8') as f:  # 打开文件
    s = f.read()  # 读取文件
    # print(s)
acc1 = []
for i in re.findall("test_accuracy:-*[0-9].[0-9]+", s):
    acc1.append(float(i.replace('test_accuracy:','')))
print(acc1)

data2 = 'unet/net37.txt'
with open(data2, "r",encoding='utf-8') as f:  # 打开文件
    s = f.read()  # 读取文件
    # print(s)
acc2 = []
for i in re.findall("test_accuracy:-*[0-9].[0-9]+", s):
    acc2.append(float(i.replace('test_accuracy:','')))
print(acc2)


plt.subplot(1,3,3)
plt.plot(x, acc1, '^g--', label="10000 training data set without Constraint", markersize=3)
plt.plot(x, acc2, 'ob-', label="10000 training data set with Constraint", markersize=3)
plt.legend(fontsize=fontsize,loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('SSIM')

ymin = 0
ymax = 1
axes = plt.gca()
axes.set_ylim([ymin,ymax])

plt.tight_layout()
plt.savefig('test_curve.png',dpi=1000)
plt.show()


