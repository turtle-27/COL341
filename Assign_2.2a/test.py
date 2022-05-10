#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

from skimage import io, transform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import sys
from tqdm import tqdm

from IPython.display import Image
# import cv2


test_file = sys.argv[1]
model_file = sys.argv[2]
pred_file = sys.argv[3]

# test_file = "../devanagari/public_test.csv"
# model_file = "model.pth"
# pred_file = "pred.txt"


# In[11]:


# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class DevanagariDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform = None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [data, labels] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,:-1].to_numpy()
            labels = data.iloc[:,-1].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
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
        image = np.array(image).astype(np.uint8).reshape(32, 32, 1)
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
#         print(image.shape, label, type(image))
        sample = {"images": image, "labels": label}
        return sample


# In[12]:


# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

# Test DataLoader
test_data = test_file # Path to test csv file
test_dataset = DevanagariDataset(data_csv = test_data, train=False, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)


# In[13]:


# Enumeration for 1 epoch
# for batch_idx, sample in enumerate(train_loader):
#     images = sample['images']
#     labels = sample['labels']


# In[14]:


class My_model(Module):   
    def __init__(self):
        super(My_model, self).__init__()
        
        self.cnn_layer1 = Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_layer2 = Sequential(
            Conv2d(32, 64, kernel_size=3, stride=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_layer3 = Sequential(
            Conv2d(64, 256, kernel_size=3, stride=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=1),
        )
        self.cnn_layer4 = Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1),
            ReLU(inplace=True),
        )
        self.linear_layers = Sequential(
            Linear(512 * 1 * 1 , 256),
            ReLU(inplace=True),
            Dropout(p = 0.2),
            Linear(256 * 1 * 1 , 46),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# In[15]:


# defining the model
model = My_model()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    
model.load_state_dict(torch.load(model_file))
model.eval()

# In[16]:


def test(model, f):
    for batch_idx, sample in enumerate(test_loader):
        x_test = sample['images']
        y_test = sample['labels'] 
        if torch.cuda.is_available():
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        pred = model(x_test)
        pred = torch.argmax(pred, dim = 1)
        for item in pred:
            f.write(str(item.item()) + "\n")
    


# In[17]:


f1 = open(pred_file, 'w')
test(model, f1)

f1.close()


