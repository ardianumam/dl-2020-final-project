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
import time, cv2


# global variable
path_to_dir = ['/Users/poyuanjeng/Documents/Deep Learning NCTU/final_project/',
               './']
path_to_dir = path_to_dir[-1]

DIR_STORE_MODEL = ['/home/austin/trailnet-testing-Pytorch/duckiefloat_line_follow/src/line_detect/src/line_angle_A.pth',
                   './model']
TRAIN_DIR = '' #modified in the code
TEST_DIR = '' #modified in the code
DIR_STORE_MODEL = DIR_STORE_MODEL[-1]

MODEL_NAME = '.pt' #specify only the extension, the name will be automatically assigned

NETWORK = ['alexnet',
           'mobilenet_v2',
           'mobilenet_v2_bg',
           'small']
NETWORK = NETWORK[1]
N_EPOCH = 200
MODE = ['train', 'infer']
MODE = MODE[1]

RESUMED_MODEL = 'model/mobilenet_v2_epoch-112.pt' # path of model you wanna resume
IS_RESUMED = True # change to True to resume the model training, vice versa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size_downscale = (75, 100) #

def main():
    global size_downscale, TRAIN_DIR, TEST_DIR

    #if(NETWORK == 'mobilenet_v2' or 'alexnet'): THISSSS OLDDDD CAUSE!!!!!
    if (NETWORK == 'alexnet'):
        size_downscale = (101,101)
    elif(NETWORK == 'mobilenet_v2'):
        size_downscale = (75,100)
        TRAIN_DIR = 'Trail_dataset18/train_data'
        TEST_DIR = 'Trail_dataset18/test_data'
    elif(NETWORK == 'mobilenet_v2_bg'):
        size_downscale = (75, 100)
        TRAIN_DIR = 'Trail_dataset19/train_data'
        TEST_DIR = 'Trail_dataset19/test_data'


    if (MODE == 'train'):
        transform0 = transforms.RandomApply([transforms.ColorJitter(),
                                                                 ],
                                            p = 0.3)
        transform = transforms.Compose([transform0, transforms.Resize(size_downscale),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        #load image from folder and set foldername as label
        train_data = datasets.ImageFolder(
            path_to_dir+TRAIN_DIR,
            transform = transform)

        test_data = datasets.ImageFolder(
            path_to_dir+TEST_DIR,
            transform = transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=40,shuffle= True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=40,shuffle=True)

        print("Used device:", DEVICE)
        print("Used network:", NETWORK)
        print("Input size:", size_downscale)
        if(NETWORK == 'alexnet'):
            net = AlexNet().to(DEVICE)
        elif(NETWORK == 'mobilenet_v2'):
            net = MobileNet2(input_size=100, num_classes=18).to(DEVICE)
        elif(NETWORK == 'small'):
            net = smallNet(n_class=18).to(DEVICE)
        elif(NETWORK == 'mobilenet_v2_bg'):
            net = MobileNet2(input_size=100, num_classes=19).to(DEVICE)
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
        print("Network:", NETWORK)
        print("Device:", DEVICE)
        IMG_IN_PATH = "./new_data_2020-1-6/01/24.jpg"
        img_cv = cv2.imread(IMG_IN_PATH, 1)
        MODEL_PATH = "./mobilenet_v2_epoch-122.pt"
        time0 = time.time()
        net = MobileNet2(input_size=100, num_classes=18).to(DEVICE)
        checkpoint = torch.load(MODEL_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        print("Load model time:", time.time()-time0)
        INPUT_TYPE = ['image', 'video', 'manual_test', 'auto_test', 'new_data_test']
        INPUT_TYPE = INPUT_TYPE[1]
        omega_array = np.array([0.1,0.17,0.24,0.305,0.37,0.44,0.505,0.73,-0.1,-0.17,-0.24,-0.305,-0.37,-0.44,-0.505,-0.73,0.0,0.0])
        if(INPUT_TYPE == 'image'):
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_cv = Image.fromarray(np.uint8(img_cv))
            img_pil = Image.open(IMG_IN_PATH)
            # img2 = list(img2.getdata())
            time0 = time.time()

            for i in range(20):
                prediction_cv = infer(net, img_cv)
                prediction_pil = infer(net, img_pil)
            print("Time predict.: ", (time.time()-time0)/20)
            print("Prediction cv: ", prediction_cv)
            print("Prediction pil: ", prediction_pil)
            # cv2.waitKey()
        elif(INPUT_TYPE == 'video'):
            vidcap = cv2.VideoCapture('./vid3.mp4')
            SHOW = ['dl', 'improc', 'both'] #improc means image processing
            SHOW = SHOW[2]
            success, image = vidcap.read()
            count = 0
            omega_dl_list = [0,0,0,0]; omega_seg_list = [0,0,0,0]
            while success:
                image = image.astype(np.uint8)
                image_bgr= np.copy(image)
                image_bgr = cv2.putText(image_bgr, "Rotation prediction (radian).", (100, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 3)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(np.uint8(image))
                if(SHOW == 'dl' or SHOW == 'both'):
                    prediction = infer(net, img)
                    if(prediction != 18):
                        omega = omega_array[prediction]
                        y = (int)(image_bgr.shape[0]/2.0)
                        x = (int)(image_bgr.shape[1]/2.0) - (int)(y*np.arctan(omega))
                        targetCoor = (x, y)
                    else:
                        omega = 0 #invalid (no robot route in the scene)
                        targetCoor = (0,0)
                    omega_dl_list.append(omega)
                    omega_dl_list.pop(0)
                    rotation_rad = (float)(sum(omega_dl_list)) / (float)(len(omega_dl_list))
                    image_bgr = cv2.putText(image_bgr, ".", (targetCoor[0], targetCoor[1]),
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 12)
                    image_bgr = cv2.putText(image_bgr, "DL: " + str(rotation_rad), (130, 140),
                                         cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 3)
                    image_bgr = cv2.line(image_bgr, ((int)(image.shape[1] / 2), (int)(image.shape[0])),
                                      targetCoor, (0, 0, 255),3)
                if(SHOW == 'improc' or SHOW == 'both'):
                    # START using segmentation approach and draw the results
                    isSegmented, targetCoor, rotation_rad = segment(image)
                    omega_seg_list.append(rotation_rad)
                    omega_seg_list.pop(0)
                    rotation_rad = (float)(sum(omega_seg_list))/(float)(len(omega_seg_list))
                    image_bgr = cv2.putText(image_bgr, ".", (targetCoor[0], targetCoor[1]),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 12)
                    image_bgr = cv2.putText(image_bgr, "Improc.: " + str(np.around(rotation_rad, decimals=4)),
                                            (130, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0), 3)
                    image_bgr = cv2.line(image_bgr, ((int)(image.shape[1] / 2), (int)(image.shape[0])),
                                       targetCoor, (0, 255, 0),3)
                    # END using segmentation approach and draw the results
                    img_seg_resize = cv2.resize(image_bgr, (640, 480), interpolation=cv2.INTER_AREA)
                cv2.imshow("Prediction result", img_seg_resize)
                count += 1
                success, image = vidcap.read()
                if cv2.waitKey(1) == 27:
                    break
        elif(INPUT_TYPE == "manual_test"):
            # read all dataset test
            TEST_DIR = "./Trail_dataset/test_data"
            wrong_pred_dir = "./wrong_pred"
            dict_class = {}
            list_folder = os.listdir(TEST_DIR); list_folder = sorted(list_folder)
            for i in range(len(list_folder)):
                dict_class[str(i)] = list_folder[i]
            list_folder = [os.path.join(TEST_DIR,i) for i in list_folder]
            list_img_path = []; class_label = []
            for idx, i in enumerate(list_folder):
                list_img = os.listdir(i); list_img = sorted(list_img)
                for j in list_img:
                    list_img_path.append(os.path.join(i,j))
                    class_label.append(idx)
            # perform inference
            sum_correct = 0; n = 0
            print("N test img:", len(class_label))
            for i,j in zip(list_img_path, class_label):
                i = i.replace("\\","/")
                dir_name = i.split('/')[-2]
                filename_ori = i.split('/')[-1]
                img = Image.open(i)
                pred = infer(net, img)
                label = j
                if(label==pred):
                    sum_correct += 1
                else: #wrong pred is encountered
                    # write wrong pred to file
                    pred_dir = dict_class[str(pred)]
                    img = np.asarray(img); img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    filename = "gt-"+dir_name+"_pred-"+pred_dir+"_"+filename_ori
                    cv2.imwrite(os.path.join(wrong_pred_dir,filename), img)
                n += 1
                if((n%100)==0):
                    print(n," -> acc:", sum_correct/n)
            print("Accumulated acc:", sum_correct / n)
        elif(INPUT_TYPE == "auto_test"):
            transform = transforms.Compose([transforms.Resize(size_downscale),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            test_data = datasets.ImageFolder(
                path_to_dir + TEST_DIR,
                transform=transform)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=40, shuffle=False)
            acc_test = test(test_loader, net)
            print("Test using pytorch loader!")
            print("acc_test:", acc_test)

            print("Dummy...")
        elif(INPUT_TYPE == 'new_data_test'):
            # read all dataset test
            TEST_DIR = "./Trail_dataset/test_data"
            wrong_pred_dir = "./wrong_pred"
            dict_class = {}
            list_folder = os.listdir(TEST_DIR);
            list_folder = sorted(list_folder)
            for i in range(len(list_folder)):
                dict_class[str(i)] = list_folder[i]
            list_folder = [os.path.join(TEST_DIR, i) for i in list_folder]
            list_img_path = [];
            class_label = []
            for idx, i in enumerate(list_folder):
                list_img = os.listdir(i);
                list_img = sorted(list_img)
                for j in list_img:
                    list_img_path.append(os.path.join(i, j))
                    class_label.append(idx)
            # perform inference
            sum_correct = 0;
            n = 0
            print("N test img:", len(class_label))
            for i, j in zip(list_img_path, class_label):
                i = i.replace("\\", "/")
                dir_name = i.split('/')[-2]
                filename_ori = i.split('/')[-1]
                img = Image.open(i)
                pred = infer(net, img)
                label = j
                if (label == pred):
                    sum_correct += 1
                else:  # wrong pred is encountered
                    # write wrong pred to file
                    pred_dir = dict_class[str(pred)]
                    img = np.asarray(img);
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    filename = "gt-" + dir_name + "_pred-" + pred_dir + "_" + filename_ori
                    cv2.imwrite(os.path.join(wrong_pred_dir, filename), img)
                n += 1
                if ((n % 100) == 0):
                    print(n, " -> acc:", sum_correct / n)
            print("Accumulated acc:", sum_correct / n)


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


class smallNet(nn.Module):
    def __init__(self, n_class):
        super(smallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(30, 40, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(37*50*40, 128)
        self.fc2 = nn.Linear(128, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

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

def segment(img):
    isSegmented = False; rotation_rad = 0; targetCoor = ((int)(img.shape[1]/2),0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Range for blue
    lower_blue = np.array([100, 100, 40])
    upper_blue = np.array([140, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # Range for yellow
    lower_yelllow = np.array([10, 100, 40])
    upper_yellow = np.array([50, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_yelllow, upper_yellow)
    mask = mask1 + mask2

    kernel = np.ones((20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("mask", mask)
    # cv2.waitKey()
    image = mask.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)

    if (nb_components >= 2): # at least has one background and segmented foreground
        areas = stats[:, -1]
        max_label = 1 #0 is for the background, so, skip it
        max_area = areas[1]
        for i in range(2, nb_components):
            if areas[i] > max_area:
                max_label = i
                max_area = areas[i]

        center = centroids[max_label].astype(np.int16)
        delta = 20
        up_lim = center[1] + delta; low_lim = center[1] - delta
        img2 = np.zeros(output.shape)
        img2[output == max_label] = 255
        crop_img = img2[low_lim:up_lim, :]

        # rerun connected component again to get the new horizontal center
        crop_img = crop_img.astype('uint8')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(crop_img, connectivity=4)
        if ((nb_components >= 2) and max_area > 700):  # at least has one background and segmented foreground
            isSegmented = True
            areas = stats[:, -1]
            max_label = 1
            max_area = areas[1]
            for i in range(2, nb_components):
                if areas[i] > max_area:
                    max_label = i
                    max_area = areas[i]
            targetCoor = ((int)(centroids[max_label][0]),(int)(center[1]))
            #calculate the rotation degree
            x_vec = (float)((img.shape[1]/2)-targetCoor[0])
            y_vec = (float)(img.shape[0]-targetCoor[1])
            rotation_rad = (float)(np.arctan(x_vec/y_vec))

    return isSegmented, targetCoor, rotation_rad

def infer(net, image):
    """
    To infer an image using trained model
    :param net: trained network where the trained parameters are already loaded
    :param image: PIL image
    :param class2deg: dictionary to map class to rotation degree
    :return: streering degree
    """
    img_pil2 = np.asarray(image)
    # print("Img PIL 2:", img_pil2)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(DEVICE)
    net.eval()
    transform = transforms.Compose([transforms.Resize(size_downscale),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    input = transform(image).to(DEVICE)
    # print("Input with manual loader:", input)
    input = input.unsqueeze(0)
    outputs = net(input)
    # print("output:", outputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted = outputs.argmax()

    return predicted.item()

main()
