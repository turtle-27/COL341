import os
import sys

import torch
import torchvision

import pandas as pd

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from skimage import io, transform

from torch.optim import SGD
from torch.nn import CrossEntropyLoss, Dropout, Flatten

import torch.nn as nn
import torch.nn.functional as F

import PIL
from PIL import Image as img

from IPython.display import Image

GLOBAL_LABEL = {}
GLOBAL_LABEL_REV = {}
	    
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, train = True, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.is_train = train
        
        if train == True:
            self.img_labels.iloc[:,1] = self.img_labels.iloc[:,1].map(GLOBAL_LABEL)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = PIL.Image.open(img_path)
        
        if self.is_train:
            label = self.img_labels.iloc[idx, 1]
        else:
            label = -1

        image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        sample = {"image": image, "label": label}
        
        return sample

def createGlobalDic(trainingFile):
    asanas = ['Virabhadrasana', 'Vrikshasana', 'Utkatasana', 'Padahastasana',
        'Katichakrasana', 'TriyakTadasana', 'Gorakshasana', 'Tadasana', 'Pranamasana',
        'ParivrittaTrikonasana', 'Tuladandasana', 'Santolanasana', 'Still',
        'Natavarasana', 'Garudasana', 'Naukasana', 'Ardhachakrasana', 'Trikonasana',
        'Natarajasana']

    for index,value in enumerate(asanas):
        GLOBAL_LABEL_REV[index] = value 
        GLOBAL_LABEL[value] = index

def loadData(file):
    BATCH_SIZE = 50 
    NUM_WORKERS = 20

    stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    valid_tfms = transforms.Compose([transforms.Resize(299),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

    root = ""

    test_dataset = CustomImageDataset(annotations_file = file, train=False, img_dir = root, transform=valid_tfms)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

    return test_loader

def modelLoader(train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CrossEntropyLoss()

#     model = models.googlenet(pretrained=True)
#     model = torchvision.models.mnasnet1_3(pretrained = False)

    model = models.inception_v3(pretrained = True)
    
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(pytorch_total_params)
    # print(model)

    model.fc = nn.Sequential( nn.Linear(model.fc.in_features, 19),)
                       
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5, nesterov = True)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,  max_lr = 0.1, epochs = 40, steps_per_epoch = len(train_dataloader))
   
    if not torch.cuda.is_available():
        return [model, criterion, optimizer, scheduler]

    model = model.cuda()
    criterion = criterion.cuda()

    return [model, criterion, optimizer, scheduler]

def main():

    modelFile = sys.argv[1]
    testFile = sys.argv[2]
    outputFile = sys.argv[3]

    createGlobalDic(testFile)

    test_loader = loadData(testFile)

    model, criterion, optimizer, scheduler = modelLoader(test_loader)
    model.load_state_dict(torch.load(modelFile))
    model.eval()

    total_count = 0
    count = 0

    predictions = []

    for batch_idx, sample in enumerate(test_loader):
        
        if not torch.cuda.is_available():
            x_test = sample['image']
            y_test = sample['label'] 
        else:
            x_test = sample['image'].cuda()
            y_test = sample['label'].cuda()

        pred = model(x_test)
        pred = torch.argmax(pred, dim = 1)

        for i in range(len(pred)):
            predictions.append(pred[i].item())

    f = open(testFile, 'r')
    filenames = f.readlines()
    f.close()

    f = open(outputFile, 'w')
    f.write('name,'+'category'+'\n')

    for i in range(len(predictions)):
        f.write(filenames[i+1][:-1] + ',' + GLOBAL_LABEL_REV[predictions[i]] + '\n')

    f.close()

main()

