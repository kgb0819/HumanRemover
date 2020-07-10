#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,models
import numpy as np
import glob
from torch.utils.data.dataset import Dataset  # For custom data-sets
import cv2

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import DataLoader
from torchsummary import summary
import pandas as pd


# In[3]:


h=320
w=240
batchSize=1
learningRate=0.0001
epoch=1


# In[4]:


class CustomDataset(Dataset):
    def __init__(self, image_paths,taget_paths, train=True):

        self.image_paths = image_paths
        self.taget_paths = taget_paths
        self.transforms = transforms.Compose([transforms.Resize((h,w)),
                                              #transforms.Grayscale(num_output_channels=3),
                                              transforms.ToTensor()])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        
        target = Image.open(self.taget_paths[index])
        
        image = self.transforms(image)
        target = self.transforms(target)
        return image, target

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)
    
def flaotTensorToImage(img, mean=0, std=1):
    """convert a tensor to an image"""
    img=img.detach().cpu().numpy()
    img[img>1.0]=1.0
    
    img = np.transpose(img, (1,2,0))
    img = np.squeeze(img)*255
    img = img.astype(np.uint8)  
    return img


# In[5]:


TestData=glob.glob("D:/Capstone/dataset/segmentation_dataset/Human-Segmentation-Dataset-master/Training_Images/*.jpg")
TagetData=glob.glob("D:/Capstone/dataset/segmentation_dataset/Human-Segmentation-Dataset-master/Ground_Truth/*.png")

TestData = np.array(TestData)
TagetData = np.array(TagetData)

test_dataset=CustomDataset(TestData,TagetData)

print(len(TestData))
print(len(TagetData))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True, num_workers=0)


# In[6]:


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        vgg16 =nn.Sequential(*list(vgg16.children())[:-2]) 
        
        for param in vgg16.parameters():
            param.require_grad = False
        
        self.vggConv1= vgg16[0][0:5]
        self.vggConv2= vgg16[0][5:10]
        self.vggConv3= vgg16[0][10:17]
        self.vggConv4= vgg16[0][17:24]
        self.vggConv5= vgg16[0][24:31]
    
        
        self.conv1=nn.Conv2d(512,1000,kernel_size=1)
        self.conv2=nn.Conv2d(1000,1000,kernel_size=1)
        self.conv3=nn.Conv2d(1000,1,kernel_size=1)
        
        self.upSample1=nn.Upsample([h//16, w//16],mode='bilinear')
        self.upSample2=nn.Upsample([h//8, w//8],mode='bilinear')
        self.upSample3=nn.Upsample([h, w],mode='bilinear')
        
        self.conv4=nn.Conv2d(512,1,kernel_size=1)
        self.conv5=nn.Conv2d(256,1,kernel_size=1)
        self.conv6=nn.Conv2d(1,1,kernel_size=1)
        
        self.bn1     = nn.BatchNorm2d(1000)
        self.bn2     = nn.BatchNorm2d(1)
        
        self.relu = nn.ReLU()
        
        
    def forward(self,image):
        
        down1=self.vggConv1(image)
        down2=self.vggConv2(down1)
        down3=self.vggConv3(down2)
        down4=self.vggConv4(down3)
        down5=self.vggConv5(down4)
        
        x2=self.relu(self.conv1(down5))
        x3=self.relu(self.conv2(x2))
        x4=self.relu(self.conv3(x3))
        
        x5=self.upSample1(x4)
        down4=self.relu(self.conv4(down4))
        x5=x5+down4
        
        x6=self.upSample2(x5)
        down3=self.relu(self.conv5(down3))
        x6=x6+down3
        
        x7=self.upSample3(x6)
        
        x8=self.relu(self.conv6(x7))
        
        return x8


# In[7]:


model=SegNet()

#print(model)

checkpoint = torch.load("./SGN_V3_1.pth")
model.load_state_dict(checkpoint['model_state_dict'])

use_cuda=1

if use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learningRate)

criterion = torch.nn.MSELoss()

summary(model, (3, h, w))


# In[9]:


for i,(image,target) in enumerate(test_loader):
    image = image.cuda(async=True)
    target = target.cuda(async=True)

    output=model(image)

    image=flaotTensorToImage(image[0])
    target=flaotTensorToImage(target[0])
    output=flaotTensorToImage(output[0])
        

    #cv2.imwrite("./data/image/"+str(i)+".jpg", image)

    #cv2.imwrite("./data/target/"+str(i)+".jpg", target)
    #cv2.imwrite("./data/output/"+str(i)+".jpg", output)

    ret3, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #cv2.imwrite("./data/threshold/"+str(i)+".jpg", thres)

    image[:,:,0]=cv2.add(image[:,:,0],output)
    image[:,:,1]=cv2.add(image[:,:,1],output)
    image[:,:,2]=cv2.add(image[:,:,2],output)
    
    plt.figure()
    plt.imshow(image)
        
    plt.figure()
    plt.imshow(target,cmap='gray')
        
    plt.figure()
    plt.imshow(output,cmap='gray')

    #cv2.imwrite("./data/newImage/"+str(i)+".jpg", image)


# In[9]:


num_epoch=1

model.train()
test_num_batches=len(test_loader)

for epoch in range(0,num_epoch):
    total_val_loss=0
    
    for i,(image,target) in enumerate(test_loader):
        
        image = image.cuda(async=True)
        target = target.cuda(async=True)
        
        optimizer.zero_grad()
        output = model(image)
        
        loss = criterion(output, target)
        
        #cv2.add(image[0][0],output[0][0])
        #cv2.add(image[0][1],output[0][0])
        #cv2.add(image[0][2],output[0][0])
        
        output = np.transpose(output[0].detach().cpu().numpy(), (1,2,0))
        output = np.squeeze(output)*255
        output = cv2.add(output,0)
        
        output = cv2.GaussianBlur(output,(5,5),0)
        

        ret3, th3 = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        target=target[0][0].detach().cpu().numpy()
        output=output[0][0].detach().cpu().numpy()
        #target=flaotTensorToImage(target[0])
        #output=flaotTensorToImage(output[0])
        
        plt.figure()
        plt.imshow(image)
        
        plt.figure()
        plt.imshow(target,cmap='gray')
        
        plt.figure()
        plt.imshow(output,cmap='gray')
        
        if i%10==0:
            print('Test [{}/{} ({}%)]\tLoss: {:.6f}'.format(i,test_num_batches,i/test_num_batches*100,loss))


# In[ ]:




