import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision.models import resnet18
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import pytorch_ssim

from model import ResNet18Unet

time_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder="D:/dataset/Fringe_colors"
target_folder="D:/dataset/Stress_maps"

epoch_lr=[(10,0.0001,1),(10,0.00001,5)]
batch_size = 128

checkpoint = 'unet/net.pth'
model_checkpoint = 'unet/net19.pth'


fringe_files_list = ['Img_' + str(i) +'.bmp' for i in range(1,100001,10)]
target_files_list = ['Target_' + str(i) +'.bmp' for i in range(1,100001,10)]
fringe_files_list1 = ['Img_' + str(i) +'.bmp' for i in range(3,100001,50)]
target_files_list1 = ['Target_' + str(i) +'.bmp' for i in range(3,100001,50)]
fringe_files=[os.path.join(data_folder,i) for i in fringe_files_list]
target_files=[os.path.join(target_folder,i) for i in target_files_list]
fringe_files1=[os.path.join(data_folder,i) for i in fringe_files_list1]
target_files1=[os.path.join(target_folder,i) for i in target_files_list1]
train_fringe_files=fringe_files
train_target_files=target_files
test_fringe_files=fringe_files1
test_target_files=target_files1


preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )
])
def default_loader(path):
    img_pil = Image.open(path)
    # img_pil = img_pil.resize((224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.images = train_fringe_files
        self.target = train_target_files
        self.loader = loader

    def __getitem__(self, index):
        fn1 = self.images[index]
        img = self.loader(fn1)
        fn2 = self.target[index]
        target = self.loader(fn2)
        return img,target

    def __len__(self):
        return len(self.images)

class testset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.images = test_fringe_files
        self.target = test_target_files
        self.loader = loader

    def __getitem__(self, index):
        fn1 = self.images[index]
        img = self.loader(fn1)
        fn2 = self.target[index]
        target = self.loader(fn2)
        return img, target

    def __len__(self):
        return len(self.images)

def train():
    net=ResNet18Unet().to(device)
    # load model params
    net.load_state_dict(torch.load(model_checkpoint)["params"])
    print('successful')

    for params in net.parameters():
        nn.init.normal_(params, mean=0, std=0.01)

    train_data = trainset()
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = testset()
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    #loss function
    criteron = nn.MSELoss()
    best_accuracy = 0
    # if os.path.exists(checkpoint):
    #     ckpt=torch.load(checkpoint)
    #     bestloss=ckpt["loss"]
    #     net.load_state_dict(ckpt["params"])
    #     print('checkpoint loaded ...')
    for n, (num_epochs, lr, ld) in enumerate(epoch_lr):
        optimizer = optim.Adam(
            net.parameters(), lr=lr, weight_decay=0,
        )
        for epoch in range(num_epochs):
            # if n == 0:
            #     ld = 1
            # else:
            #     ld = 1 + epoch*0.2
            net.eval()
            epoch_loss = 0.0
            for i, (img, target) in enumerate(trainloader):
                out = net(img.to(device))
                # print(out.shape)
                # print(target.shape)
                ssim_loss = 1 - pytorch_ssim.ssim(out, target.to(device).float())
                out = out.squeeze(1)
                target = target.to(device).float().squeeze(1)
                if n == 1 or n == 0:
                    physics_loss = 0.0
                    for j in range(out.shape[0]):
                        batch_t = out[j]
                        for p in range(epoch+1,batch_t.shape[0]-1,10):
                            for q in range(epoch+1,batch_t.shape[1]-1,10):
                                batch_pq_ave = (batch_t[p-1][q]+batch_t[p+1][q]+batch_t[p][q-1]+batch_t[p][q+1])/4
                                physics_loss += (batch_t[p][q]-batch_pq_ave)*(batch_t[p][q]-batch_pq_ave)
                    loss = ssim_loss + 100 * criteron(out, target) + 0.0001 * physics_loss
                    # print(loss, ssim_loss, physics_loss, criteron(out, target))

                # loss = ssim_loss + ld * criteron(out, target)
                # loss = 1 - pytorch_ssim.ssim(out,target.to(device).float()) + ld * criteron(out.squeeze(1),target.to(device).float().squeeze(1))
                # print(out.squeeze(1))
                # print(target.to(device).float().squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            #test
            with torch.no_grad():
                net.eval()
                test_accuracy = 0.0
                batch = 0
                for i, (img, target) in enumerate(testloader):
                    out = net(img.to(device))
                    loss = pytorch_ssim.ssim(out, target.to(device).float())
                    batch += 1
                    test_accuracy += loss.item()
                # print loss
                print("test_accuracy:{}".format(test_accuracy / batch))
                time_end = time.time()
                print('totally cost', time_end - time_start)
            if test_accuracy / batch > best_accuracy:
                best_accuracy = test_accuracy / batch
                torch.save(
                    {"params":net.state_dict(), "accuracy":test_accuracy}, checkpoint
                )
                print("model save")

if __name__ == "__main__":
    train()












