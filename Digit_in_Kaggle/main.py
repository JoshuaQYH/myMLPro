import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from ConvModel import ConvNet
import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers

import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cpu')

from MNIST_data import MNIST_data



# batch_size, epoch and iteration
batch_size = 64
n_iters = 2500
learning_rate = 0.01



# load data
train_dataset = MNIST_data("train.csv", transform= transforms.Compose(
                            [transforms.ToPILImage(), 
                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_dataset = MNIST_data("test.csv")
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle = True)


# show example
ConvModel = ConvNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(ConvModel.parameters(), lr = learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


def train(num_epochs):
    ConvModel.train()
    exp_lr_scheduler.step()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            samples = Variable(images).to(device)
            labels = Variable(labels).to(device)

            output = ConvModel(samples.reshape(-1, 1, 28, 28))
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print("Epoch:{}/{}, step:{}, loss:{:.4f}".format(epoch+1, num_epochs, i + 1, loss.item()))

def evaluate(data_loader):
    ConvModel.eval()
    loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = Variable(data), Variable(target)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            output = ConvModel(data.reshape(-1, 1, 28, 28))
            
            loss += criterion(output, target).item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
        
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

for i in range(2):
    train(1)
    evaluate(train_loader)

def prediciton(data_loader):
    ConvModel.eval()
    test_pred = torch.LongTensor()
    for i, data in enumerate(data_loader):
        data = Variable(data)
        if torch.cuda.is_available():
            data = data.cuda()
        output = ConvModel(data.reshape(-1, 1, 28, 28))
        pred = output.data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
    return test_pred


test_pred = prediciton(test_loader)
out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1)[:,None], test_pred.numpy()], 
                      columns=['ImageId', 'Label'])
out_df.to_csv("sample_submission.csv", sep = ',')