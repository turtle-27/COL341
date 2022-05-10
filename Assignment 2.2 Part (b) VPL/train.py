#!/usr/bin/env python
# coding: utf-8

# In[26]:


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

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import os
import sys
from tqdm import tqdm

from IPython.display import Image
# import cv2


train_file = sys.argv[1]
test_file = sys.argv[2]
model_file = sys.argv[3]
loss_file = sys.argv[4]
accuracy_file = sys.argv[5]

# train_file = "CIFAR/train_data.csv"
# test_file = "CIFAR/public_test.csv"
# model_file = "model_cifar.pth"
# loss_file = "train_loss_b.txt"
# accuracy_file = "test_acc_b.txt "


# In[27]:


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


# In[28]:


# Data Loader Usage

BATCH_SIZE = 200 # Batch Size. Adjust accordingly
NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

# Train DataLoader
train_data = train_file # Path to train csv file
# train_data = "/mnt/scratch1/siy197580/COL341/cifar/train_data.csv"
# test_data = "/mnt/scratch1/siy197580/COL341/cifar/public_test.csv"
train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_transforms)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

# Test DataLoader
test_data = test_file # Path to test csv file
# train_data = "/mnt/scratch1/siy197580/COL341/cifar/train_data.csv"
# test_data = "/mnt/scratch1/siy197580/COL341/cifar/public_test.csv"
test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=img_transforms)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)


# In[29]:


# Enumeration for 1 epoch
# for batch_idx, sample in enumerate(train_loader):
#     images = sample['images']
#     labels = sample['labels']


# In[30]:


class My_model(Module):   
    def __init__(self):
        super(My_model, self).__init__()
        
        self.cnn_layer1 = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1),
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
            Conv2d(64, 512, kernel_size=3, stride=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_layer4 = Sequential(
            Conv2d(512, 1024, kernel_size=2, stride=1),
            ReLU(inplace=True),
        )
        self.linear_layers = Sequential(
            Linear(1024 * 1 * 1 , 256),
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
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# In[31]:


# defining the model
model = My_model()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=1e-4)
# defining the loss function
Entropy = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    Entropy = Entropy.cuda()
    
# print(model)


# In[32]:


#Training
n_epochs = 5
train_loss = []
# training the model
plot_X = []
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
    print('Epoch : ',epoch+1, '\t', 'accuracy :', count/total_count)
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
        tr_loss = loss_train.item()
        total_loss += tr_loss
        count += 1
    # printing the training loss
    print('Epoch : ',epoch+1, '\t', 'loss :', total_loss/count)
    train_loss.append(total_loss/count)
    
    test(epoch)
    


# In[33]:


for epoch in range(n_epochs):
    train(epoch)
    plot_X.append(epoch+1)
torch.save(model.state_dict(), model_file)
f1 = open(loss_file, 'w')
for loss in train_loss:
    f1.write(str(loss) + "\n")
f1.close()


# In[34]:


#Plot
plt.xlabel("Epochs")
plt.ylabel("training_loss")
plt.plot(plot_X, train_loss, marker = 'o')
plt.title("Training_loss vs Epochs")
plt.grid()
plt.savefig("loss.png")
plt.close()


# In[35]:


f2 = open(accuracy_file, 'w')
for acc in test_accu:
    f2.write(str(acc) + "\n")
f2.close()

#Plot
plt.xlabel("Epochs")
plt.ylabel("test_accuracy")
plt.plot(plot_X, test_accu, marker = 'x', color = 'g')
plt.title("Test_accuracy vs Epochs")
plt.grid()
plt.savefig("accuracy.png")
plt.close()

