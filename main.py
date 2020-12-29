#! /usr/bin/env python

import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision, os
from PIL import Image
from mobilenet_v2 import MobileNet2

# global variable
path_to_dir = ['/Users/poyuanjeng/Documents/Deep Learning NCTU/final_project/',
               './']
path_to_dir = path_to_dir[-1]

DIR_STORE_MODEL = ['/home/austin/trailnet-testing-Pytorch/duckiefloat_line_follow/src/line_detect/src/line_angle_A.pth',
                   './model']
DIR_STORE_MODEL = DIR_STORE_MODEL[-1]

MODEL_NAME = '.pt' #specify only the extension, the name will be automatically assigned

NETWORK = ['alexnet',
           'mobilenet_v2']
NETWORK = NETWORK[1]
N_EPOCH = 100
MODE = ['train', 'infer']
MODE = MODE[1]

RESUMED_MODEL = 'model/mobilenet_v2_epoch-60.pt' # path of model you wanna resume
IS_RESUMED = True # change to True to resume the model training, vice versa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if (MODE == 'train'):
        transform = transforms.Compose([transforms.Resize((101,101)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        #load image from folder and set foldername as label
        train_data = datasets.ImageFolder(
            path_to_dir+'Trail_dataset/train_data',
            transform = transform)

        test_data = datasets.ImageFolder(
            path_to_dir+'Trail_dataset/test_data',
            transform = transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=40,shuffle= True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=40,shuffle=True)

        print("Used device:", DEVICE)
        print("Used network:", NETWORK)
        if(NETWORK == 'alexnet'):
            net = AlexNet().to(DEVICE)
        elif(NETWORK == 'mobilenet_v2'):
            net = MobileNet2(input_size=101, num_classes=18).to(DEVICE)
        else:
            raise Exception("Whoops! Select the correct network or make the new one.")

        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        epoch_last = 0
        if(IS_RESUMED):
            print("--- RESUMING THE TRAINING ----")
            checkpoint = torch.load(RESUMED_MODEL)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_last = checkpoint['epoch']
            loss_last = checkpoint['loss']
            print("Last resumed epoch:", epoch_last)
            print("Last resumed model loss:", loss_last)

        # make model directory if doesn't exist yet
        if(not os.path.exists(DIR_STORE_MODEL)):
            os.makedirs(DIR_STORE_MODEL)

        for epoch in range(N_EPOCH-epoch_last):  # loop over the dataset multiple times
            epoch = epoch + epoch_last + 1
            running_loss_batch = 0.0; running_loss_mini = 0.0; batch_idx = 0
            for i, data in enumerate(train_loader, 0):
                i += 1; batch_idx += 1
                # get the input
                inputs, labels = data

                # wrap time in Variable
                inputs, labels = Variable(inputs).to(DEVICE), Variable(labels).to(DEVICE)

                # zero the parameter gradients
                net.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss_batch += loss.item()
                running_loss_mini += loss.item()
                if i % 10 == 0:  # print every 'x' batches
                    print('Epoch-%d, iter-%d --> loss: %.3f' %
                          (epoch, i, running_loss_mini / 10))
                    running_loss_mini = 0

                if i % 40 == 0:  # calculate acc on test data every 'x' batches
                    acc_test = test(test_loader, net)
                    net.train() #don't forget to put it back in training mode
                    print("--------------------------------")
                    print("Test acc iter-"+str(i)+": "+str(acc_test)+"%")
                    print("--------------------------------")
            if ((epoch % 1) == 0): # save model every 'x' epoch
                MODEL_NAME2 = MODEL_NAME.replace('.pt', NETWORK+'_epoch-' + str(epoch) + ".pt")
                print('Model in ' + str(epoch) + " is stored.")
                torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': (running_loss_batch/batch_idx)},
                           os.path.join(DIR_STORE_MODEL, MODEL_NAME2))
    elif(MODE == 'infer'):
        IMG_IN_PATH = "./Trail_dataset/test_data/L_1/0.1082408_529.jpeg"
        MODEL_PATH = "./model/mobilenet_v2_epoch-80.pt"
        net = MobileNet2(input_size=101, num_classes=18)
        checkpoint = torch.load(MODEL_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        img = Image.open(IMG_IN_PATH)
        prediction = infer(net, img)
        print("Prediction: ", prediction)

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
        return F.log_softmax(x, dim=1)

#Accuracy present
def test(loader, net):
    net.eval() # set the network mode to evaluation mode
    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
             outputs = net(Variable(images))
             _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return (100.0 * correct / total).item()

def infer(net, image):
    """
    To infer an image using trained model
    :param net: trained network where the trained parameters are already loaded
    :param image: PIL image
    :param class2deg: dictionary to map class to rotation degree
    :return: streering degree
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(DEVICE)
    net.eval()
    transform = transforms.Compose([transforms.Resize((101, 101)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    input = transform(image).to(DEVICE)
    input = input.unsqueeze(0)
    outputs = net(input)
    _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

main()
