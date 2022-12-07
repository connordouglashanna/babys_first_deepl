# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 01:39:17 2022

@author: condo
"""

#%% setup

# importing pytorch et al
import torch
import torchvision

# device config
# this defines a global setting for Torch so that it knows what hardware
# to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

#%% data procurement

# importing data packages
from torchvision import datasets
from torchvision.transforms import ToTensor

# using the MNIST function to import data from the API
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
    )

test_data = datasets.MNIST(
    root = 'data', 
    train = False,
    transform = ToTensor()
    )

# inspecting the resulting object characteristics
print(train_data)
print(test_data)

# inspecting the object size
print(train_data.data.size())
print(test_data.data.size())

## rudimentary data visualization

# importing matplotlib
import matplotlib.pyplot as plt

# plotting the first entry
plt.imshow(train_data.data[0], cmap ='gray')
# note that unlike ggplot because Python stores the results of functions by default 
# these lines will generate objects which are synthesized by plt.show() 
plt.title('%i' % train_data.targets[0])
plt.show()

# plotting multiple entries
figure = plt.figure(figsize = (10, 8))
cols, rows = 5, 5
# note the way that Python assigns multiple objects in a single line using commas
for i in range(1, cols * rows + 1):
    # iterating across the matrix?
    sample_idx = torch.randint(len(train_data), size = (1,)).item()
    # defining the sample index as a random index from the training sample
    # this uses tensors, a k*m*n cube of matrices
    img, label = train_data[sample_idx]
    # this object is bizarre
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = "gray")
plt.show()

## Scrubadub dub

# importing torch utilities
from torch.utils.data import DataLoader

loaders = {
    'train' : torch.utils.data.DataLoader(train_data,
                                          batch_size = 100,
                                          shuffle = True,
                                          num_workers = 0),
    
    'test' : torch.utils.data.DataLoader(test_data, 
                                         batch_size = 100,
                                         shuffle = True,
                                         num_workers = 0),
    }
# this gives us another object layer which instructs a future function 
# to iterate over the dataset with a given sample plan
loaders

#%% PREPARE OBJECT DEFINITIONS
# using a convolutional neural network

# loading the torch subpackage for the convolutional nn algo
import torch.nn as nn

# defining the model object class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16, 
                kernel_size = 5,
                stride = 1,
                padding = 2,
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # flattening the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x # now we have x for visualization

# for detailed documentation see: 
# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

# inspecting the resulting object
cnn = CNN()
print(cnn)

# defining our loss function
loss_func = nn.CrossEntropyLoss()
loss_func

# importing torch optimization toolkit
from torch import optim 

# defining our optimization function
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
optimizer



#%% PREPARE TO TRAIN

# importing training toolkit
from torch.autograd import Variable

# how many times will our model iterate over the training data?
num_epochs = 10 

# manually defining our training function, because apparently we need to
def train(num_epochs, cnn, loaders):
    
    #assigning alias?
    cnn.train()
    
    # TRAIN BBY TRAIN
    total_step = len(loaders['train'])
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            # I have no idea what the fuck this means
            b_x = Variable(images) 
            b_y = Variable(labels)
            
            # note, study indentation flow in Python
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, 
                               total_step, loss.item()))               
# plot log scale losses using the loss.item
# to do this store those in a list
# this will show accuracy/loss per block 
# possibly facet this?


# why is my code breaking here when I don't run these blocks one at a time?

train(num_epochs, cnn, loaders)

#%% TESTING TESTING 1-2-3

# defining the test function
def test():
    
    # assigning alias
    cnn.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item()
            float(labels.size(0))
    print('Test accuracy of the model on the 10,000 test images: %.2f' % accuracy)

test()        
