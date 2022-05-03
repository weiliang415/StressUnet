import pandas as pd
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils
from PIL import Image
import pytorch_ssim
import numpy as np
import math
import cv2

from model import ResNet18Unet
checkpoint = 'unet/net21.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder="E:/zwl/北航学习/科研/ljb/项目/dataset/Fringe_colors"
target_folder="E:/zwl/北航学习/科研/ljb/项目/dataset/Stress_maps"


net=ResNet18Unet().to(device)
net.load_state_dict(torch.load(checkpoint)["params"])
net.eval()


# ring 13141 13156 13171 13186 13201 13216
# img1 = os.path.join(data_folder,"Img_13141.bmp")
# target_img1 = os.path.join(target_folder,"Target_13141.bmp")
# img2 = os.path.join(data_folder,"Img_13156.bmp")
# target_img2 = os.path.join(target_folder,"Target_13156.bmp")
# img3 = os.path.join(data_folder,"Img_13171.bmp")
# target_img3 = os.path.join(target_folder,"Target_13171.bmp")
# img1 = os.path.join(data_folder,"Img_13186.bmp")
# target_img1 = os.path.join(target_folder,"Target_13186.bmp")
# img2 = os.path.join(data_folder,"Img_13201.bmp")
# target_img2 = os.path.join(target_folder,"Target_13201.bmp")
# img3 = os.path.join(data_folder,"Img_13216.bmp")
# target_img3 = os.path.join(target_folder,"Target_13216.bmp")

# bunny = 99991 100006 100021 100036 100051 100066
# img1 = os.path.join(data_folder,"Img_99991.bmp")
# target_img1 = os.path.join(target_folder,"Target_99991.bmp")
# img2 = os.path.join(data_folder,"Img_100006.bmp")
# target_img2 = os.path.join(target_folder,"Target_100006.bmp")
# img3 = os.path.join(data_folder,"Img_100021.bmp")
# target_img3 = os.path.join(target_folder,"Target_100021.bmp")
# img1 = os.path.join(data_folder,"Img_100036.bmp")
# target_img1 = os.path.join(target_folder,"Target_100036.bmp")
# img2 = os.path.join(data_folder,"Img_100051.bmp")
# target_img2 = os.path.join(target_folder,"Target_100051.bmp")
# img3 = os.path.join(data_folder,"Img_100066.bmp")
# target_img3 = os.path.join(target_folder,"Target_100066.bmp")

# dragon 100351,100366,100381,100396,100411,100426
# img1 = os.path.join(data_folder,"Img_100351.bmp")
# target_img1 = os.path.join(target_folder,"Target_100351.bmp")
# img2 = os.path.join(data_folder,"Img_100366.bmp")
# target_img2 = os.path.join(target_folder,"Target_100366.bmp")
# img3 = os.path.join(data_folder,"Img_100381.bmp")
# target_img3 = os.path.join(target_folder,"Target_100381.bmp")
# img1 = os.path.join(data_folder,"Img_100396.bmp")
# target_img1 = os.path.join(target_folder,"Target_100396.bmp")
# img2 = os.path.join(data_folder,"Img_100411.bmp")
# target_img2 = os.path.join(target_folder,"Target_100411.bmp")
# img3 = os.path.join(data_folder,"Img_100426.bmp")
# target_img3 = os.path.join(target_folder,"Target_100426.bmp")

# img1 = os.path.join(data_folder,"Img_26066.bmp")
# target_img1 = os.path.join(target_folder,"Target_26066.bmp")
# img2 = os.path.join(data_folder,"Img_27066.bmp")
# target_img2 = os.path.join(target_folder,"Target_27066.bmp")
# img3 = os.path.join(data_folder,"Img_26585.bmp")
# target_img3 = os.path.join(target_folder,"Target_26585.bmp")


preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )
])



def psnr1(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse), mse

def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)), mse*255*255

def calculate_ssim(photo_list,name):
    print(name)
    for i in photo_list:
        img_name = 'Img_' + str(i) + '.bmp'
        target_name = 'Target_' + str(i) + '.bmp'
        img_c = os.path.join(data_folder, img_name)
        target_img_c = os.path.join(target_folder,target_name)
        img_pil = Image.open(img_c)
        img_tensor = preprocess(img_pil)
        predict = net(img_tensor.unsqueeze(0).to(device)) * 255
        print(pytorch_ssim.ssim(predict / 255, preprocess(Image.open(target_img_c)).unsqueeze(0)))

def calculate_psnr_mse(photo_list,name):
    print(name)
    for i in photo_list:
        img_name = 'Img_' + str(i) + '.bmp'
        target_name = 'Target_' + str(i) + '.bmp'
        img_c = os.path.join(data_folder, img_name)
        target_img_c = os.path.join(target_folder,target_name)

        img_pil = Image.open(img_c)
        img_tensor = preprocess(img_pil)
        predict = net(img_tensor.unsqueeze(0).to(device)) * 255

        img_1 = np.array(Image.open(target_img_c))
        img_2 = predict.squeeze(0).squeeze(0).detach().numpy()
        for p in range(1,int(img_2.shape[0]/2)):
            for n in range(p,img_2.shape[1]-p-1):
                m = p
                img_2[m][n] = (img_2[m-1][n]+img_2[m+1][n]+img_2[m][n-1]+img_2[m][n+1])/4
            for m in range(p,img_2.shape[0]-p-1):
                n = img_2.shape[0]-p-1
                img_2[m][n] = (img_2[m-1][n]+img_2[m+1][n]+img_2[m][n-1]+img_2[m][n+1])/4
            for n in range(img_2.shape[1]-p-1, p, -1):
                m = img_2.shape[0]-p-1
                img_2[m][n] = (img_2[m-1][n]+img_2[m+1][n]+img_2[m][n-1]+img_2[m][n+1])/4
            for m in range(img_2.shape[0]-p-1, p, -1):
                n = p
                img_2[m][n] = (img_2[m-1][n]+img_2[m+1][n]+img_2[m][n-1]+img_2[m][n+1])/4
        print(psnr1(img_1, img_2))

ring = [13141, 13156, 13171, 13186, 13201, 13216]
print(calculate_ssim(ring,'ring'))
print(calculate_psnr_mse(ring,'ring'))
bunny = [99991, 100006, 100021, 100036, 100051, 100066]
print(calculate_ssim(bunny,'bunny'))
print(calculate_psnr_mse(bunny,'bunny'))
dragon = [100351,100366,100381,100396,100411,100426]
print(calculate_ssim(dragon,'dragon'))
print(calculate_psnr_mse(dragon,'dragon'))
frog = [100712, 100727, 100742, 100757, 100772, 100787]
print(calculate_ssim(frog,'frog'))
print(calculate_psnr_mse(frog,'frog'))
dragon_2 = [101073, 101088, 101103, 101118, 101133, 101148]
print(calculate_ssim(dragon_2,'dragon_2'))
print(calculate_psnr_mse(dragon_2,'dragon_2'))


img_pil1 = Image.open(img1)
img_tensor1 = preprocess(img_pil1)
target_img_pil1 = Image.open(target_img1)
# t = np.array(target_img_pil1)
# for i in range(1,223):
#     for j in range(1,223):
#         ct = np.average([t[i-1][j] , t[i+1][j] , t[i][j-1] , t[i][j+1]])
#         if abs(t[i][j]-ct) > 1.25:
#             print(i,j,t[i][j],ct,t[i-1][j],t[i+1][j],t[i][j-1],t[i][j+1])
# print(np.array(target_img_pil1)[220:225,145:150])
predict1 = net(img_tensor1.unsqueeze(0).to(device))*255
predict_img1 = predict1.long().squeeze(0)
print(pytorch_ssim.ssim(predict1/255,preprocess(Image.open(target_img1)).unsqueeze(0)))
img_pil2 = Image.open(img2)
img_tensor2 = preprocess(img_pil2)
target_img_pil2 = Image.open(target_img2)
predict2 = net(img_tensor2.unsqueeze(0).to(device))*255
predict_img2 = predict2.long().squeeze(0)
print(pytorch_ssim.ssim(predict2/255,preprocess(Image.open(target_img2)).unsqueeze(0)))
img_pil3 = Image.open(img3)
img_tensor3 = preprocess(img_pil3)
target_img_pil3 = Image.open(target_img3)
predict3 = net(img_tensor3.unsqueeze(0).to(device))*255
predict_img3 = predict3.long().squeeze(0)
print(pytorch_ssim.ssim(predict3/255,preprocess(Image.open(target_img3)).unsqueeze(0)))
plt.show()

# plt.subplot(331)
# plt.imshow(img_pil1)
# plt.subplot(332)
# plt.imshow(target_img_pil1,cmap=plt.cm.gray,vmin=0,vmax=255)
# plt.subplot(333)
# plt.imshow(predict_img1.data.cpu().numpy().squeeze(0),cmap=plt.cm.gray,vmin=0,vmax=255)
# plt.subplot(334)
# plt.imshow(img_pil2)
# plt.subplot(335)
# plt.imshow(target_img_pil2,cmap=plt.cm.gray,vmin=0,vmax=255)
# plt.subplot(336)
# plt.imshow(predict_img2.data.cpu().numpy().squeeze(0),cmap=plt.cm.gray,vmin=0,vmax=255)
# plt.subplot(337)
# plt.imshow(img_pil3)
# plt.subplot(338)
# plt.imshow(target_img_pil3,cmap=plt.cm.gray,vmin=0,vmax=255)
# plt.subplot(339)
# plt.imshow(predict_img3.data.cpu().numpy().squeeze(0),cmap=plt.cm.gray,vmin=0,vmax=255)
# plt.savefig('res.png',dpi=600,bbox_inches='tight')
# plt.show()









