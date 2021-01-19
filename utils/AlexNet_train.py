#! /usr/bin/env python

import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision

path_to_dir = '/Users/poyuanjeng/Documents/Deep Learning NCTU/final_project/'

#load image from folder and set foldername as label
train_data = datasets.ImageFolder(
    path_to_dir+'Trail_dataset/train_data',
    transform = transforms.Compose([transforms.Resize((101,101)), transforms.ToTensor()])
)

test_data = datasets.ImageFolder(
    path_to_dir+'Trail_dataset/test_data',
    transform = transforms.Compose([transforms.Resize((101,101)), transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=40,shuffle= True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=40,shuffle=True)

#CNN model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 18),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = AlexNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(60): # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the input
        inputs, labels = data

        # wrap time in Variable
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:   # print every 2000 mini-batches
            print('[%d, %d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(),'/home/austin/trailnet-testing-Pytorch/duckiefloat_line_follow/src/line_detect/src/line_angle_A.pth')

#Accuracy present
print('Accuracy testing...')
correct = 0
total = 0
for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
         outputs = net(Variable(images))
         _,predicted = torch.max(outputs.data,1)
    #print('predict:',predicted)
    #print('labels.:',labels)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
