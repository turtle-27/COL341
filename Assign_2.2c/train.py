#!/usr/bin/env python
# coding: utf-8

# In[309]:


import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Dropout2d
from torch.optim import Adam, SGD, Adamax,  ASGD, LBFGS, AdamW

from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

from IPython.display import Image
# import cv2


train_file = sys.argv[1]
test_file = sys.argv[2]
model_file = sys.argv[3]
loss_file = sys.argv[4]
accuracy_file = sys.argv[5]
start = time.time()
# train_file = "../CIFAR/train_data.csv"
# test_file = "../CIFAR/public_test.csv"
# model_file = "model_cifar.pth"
# loss_file = "train_loss_b.txt"


# In[310]:


# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [labels,data] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:]
            labels = None
        
        self.images = images
        self.labels = labels
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32,32,3), order="F")


        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample


# In[311]:


# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

#norm1 = ((0,0,0), (1,1,1))
norm2 = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
norm3 = ((0.5,0.5,0.5), (0.5,0.5,0.5))


img_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(*norm2)])

img_tran = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding = 4, padding_mode='reflect'), transforms.RandomHorizontalFlip(),  transforms.ToTensor(), transforms.Normalize(*norm2,inplace=True)])

# img_tran1 = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
#                               transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
#                               transforms.RandomRotation(10),     #Rotates the image to a specified angel
#                               transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
#                               transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
#                               transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
#                                ])

# img_tran2 = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding = 4, padding_mode='reflect'),  #resises the image so it can be perfect for our model.
#                               transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
#                               transforms.RandomRotation(10),     #Rotates the image to a specified angel
#                               transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
#                               transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
#                               transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
#                               transforms.Normalize(*norm2,inplace=True) #Normalize all the images
#                                ])

# Train DataLoader
train_data = train_file # Path to train csv file
# train_data = "/mnt/scratch1/siy197580/COL341/cifar/train_data.csv"
# test_data = "/mnt/scratch1/siy197580/COL341/cifar/public_test.csv"
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_tran)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
test_data = test_file # Path to test csv file
# train_data = "/mnt/scratch1/siy197580/COL341/cifar/train_data.csv"
# test_data = "/mnt/scratch1/siy197580/COL341/cifar/public_test.csv"
test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)


# In[312]:


# Enumeration for 1 epoch
# for batch_idx, sample in enumerate(train_loader):
#     images = sample['images']
#     labels = sample['labels']


# In[321]:


class My_model(Module):   
    def __init__(self):
        super(My_model, self).__init__()
        
        self.cnn_layer1 = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            #MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_layer2 = Sequential(
            Conv2d(32, 64, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_layer3 = Sequential(
            Conv2d(64, 64, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(64),
            ReLU(inplace=True),
        )
        self.cnn_layer4 = Sequential(
            Conv2d(64, 64, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(64),
            ReLU(inplace=True),
        )
        self.cnn_layer5 = Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_layer6 = Sequential(
            Conv2d(128, 128, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(128),
            ReLU(inplace=True),
        )
        
        self.cnn_layer7 = Sequential(
            Conv2d(128, 256, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),  
        )
        self.cnn_layer8 = Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding = 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(
            Linear(512 * 2 * 2 , 256),
            ReLU(inplace=True),
            Dropout(p = 0.2),
            Linear(256 * 1 * 1 , 10),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = self.cnn_layer5(x)
        x = self.cnn_layer6(x)
        x = self.cnn_layer7(x)
        x = self.cnn_layer8(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# In[322]:


# class My_model(Module):   
#     def __init__(self):
#         super(My_model, self).__init__()
        
#         self.cnn_layer1 = Sequential(
#             Conv2d(3, 32, kernel_size=3, stride=1),
#             BatchNorm2d(32),
#             ReLU(inplace=True),
#         )
#         self.cnn_layer2 = Sequential(
#             Conv2d(32, 64, kernel_size=3, stride=1),
#             BatchNorm2d(64),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.cnn_layer3 = Sequential(
#             Conv2d(64, 128, kernel_size=3, stride=1),
#             BatchNorm2d(128),
#             ReLU(inplace=True),
#         )
#         self.cnn_layer4 = Sequential(
#             Conv2d(128, 128, kernel_size=3, stride=1),
#             BatchNorm2d(128),
#             ReLU(inplace=True),
#         )
#         self.cnn_layer5 = Sequential(
#             Conv2d(128, 128, kernel_size=3, stride=1),
#             BatchNorm2d(128),
#             ReLU(inplace=True),
#         )
#         self.cnn_layer6 = Sequential(
#             Conv2d(128, 512, kernel_size=3, stride=1),
#             BatchNorm2d(512),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.cnn_layer7 = Sequential(
#             Conv2d(512, 1024, kernel_size=3, stride=1),
#             ReLU(inplace=True),
#         )
#         self.linear_layers = Sequential(
#             Linear(1024 * 1 * 1 , 256),
#             ReLU(inplace=True),
#             Dropout(p = 0.2),
#             Linear(256 * 1 * 1 , 10),
#         )

#     # Defining the forward pass    
#     def forward(self, x):
#         x = self.cnn_layer1(x)
#         x = self.cnn_layer2(x)
#         x = self.cnn_layer3(x)
#         x = self.cnn_layer4(x)
#         x = self.cnn_layer5(x)
#         x = self.cnn_layer6(x)
#         x = self.cnn_layer7(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x


# In[323]:


# class My_model(Module):   
#     def __init__(self):
#         super(My_model, self).__init__()
        
#         self.cnn_layer1 = Sequential(
#             Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             BatchNorm2d(32),
#             ReLU(inplace=True),
#             Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.cnn_layer2 = Sequential(
#             Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             BatchNorm2d(128),
#             ReLU(inplace=True),
#             Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#             Dropout2d(p=0.05),
#         )
#         self.cnn_layer3 = Sequential(
#             Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             BatchNorm2d(256),
#             ReLU(inplace=True),
#             Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.linear_layers = Sequential(
#             Dropout(p=0.1),
#             Linear(4096, 1024),
#             ReLU(inplace=True),
#             Linear(1024, 512),
#             ReLU(inplace=True),
#             Dropout(p=0.1),
#             Linear(512, 10)
#         )

#     # Defining the forward pass    
#     def forward(self, x):
#         x = self.cnn_layer1(x)
#         x = self.cnn_layer2(x)
#         x = self.cnn_layer3(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x


# In[324]:


# defining the model
# accuracy_file = "final_acc.txt"
model = My_model()
# defining the optimizer
optimizer = SGD(model.parameters(), lr=0.1, momentum = 0.9, nesterov = True)
# defining the loss function
Entropy = CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    Entropy = Entropy.cuda()
    
# print(model)


# In[325]:


#Training
n_epochs = 50
train_loss = []
# training the model
# plot_X = []
test_accu = []

def test(epoch):
    model.eval()
    total_count = 0
    count = 0
    for batch_idx, sample in enumerate(test_loader):
        x_test = sample['images']
        y_test = sample['labels'] 
        if torch.cuda.is_available():
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        pred = model(x_test)
        pred = torch.argmax(pred, dim = 1)
        for i in range(len(y_test)):
            if(y_test[i] == pred[i]):
                count += 1
            total_count += 1
    # print('Epoch : ',epoch+1, '\t', 'accuracy :', count/total_count)
    test_accu.append(count/total_count)

def train(epoch):
    model.train()
    tr_loss = 0
    total_loss = 0
    count = 0
    for batch_idx, sample in enumerate(train_loader):
        x_train = sample['images']
        y_train = sample['labels'] 
        if torch.cuda.is_available():
          x_train = x_train.cuda()
          y_train = y_train.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)

        # computing the training and validation loss
        loss_train = Entropy(output_train, y_train)
        
        
        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        tr_loss = loss_train.item()
        total_loss += tr_loss
        count += 1
    # printing the training loss
    # print('Epoch : ',epoch+1, '\t', 'loss :', total_loss/count)
    train_loss.append(total_loss/count)
    
    test(epoch)


# In[326]:


pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
for epoch in range(n_epochs):
    if((time.time() - start) > 1700):
        break
    train(epoch)
    
    # plot_X.append(epoch+1)
torch.save(model.state_dict(), model_file)
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)
f1 = open(loss_file, 'w')
for loss in train_loss:
    f1.write(str(loss) + "\n")
f1.close()


# In[327]:


#Plot
# plt.xlabel("Epochs")
# plt.ylabel("training_loss")
# plt.plot(plot_X, train_loss, marker = 'o')
# plt.title("Training_loss vs Epochs")
# plt.grid()
# plt.savefig("part2_train_loss.png")


# In[328]:


f2 = open(accuracy_file, 'w')
for acc in test_accu:
    f2.write(str(acc) + "\n")
f2.close()

#Plot
# plt.xlabel("Epochs")
# plt.ylabel("test_accuracy")
# plt.plot(plot_X, test_accu, marker = 'x', color = 'g')
# plt.title("Test_accuracy vs Epochs")
# plt.grid()
# plt.savefig("part2_test_acc.png")

