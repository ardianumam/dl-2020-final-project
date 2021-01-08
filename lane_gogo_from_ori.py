#!/usr/bin/env python
from __builtin__ import True
import numpy as np
import rospy
import math
import torch
import rospkg
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torchvision
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
import cv2
import os
from torchvision import transforms, utils, datasets
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from PIL import Image as ImagePIL
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

os.environ['ROS_IP'] = '10.42.0.1'
bridge = CvBridge()
# weight_path = '/home/sis/ncsist_threat_processing/trailnet-testing-Pytorch/2020_summer/src/deep_learning/src/yb_lane.pth'
weight_path = os.path.join(os.getcwd(), "mobilenet_v2_epoch-122.pt")

'''
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(34048, 200)
        self.fc2 = nn.Linear(200, 18)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
'''
# to define mobileNet_v2 class
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class MobileNet2(nn.Module):
    """MobileNet2 implementation.
    """

    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU6):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """

        super(MobileNet2, self).__init__()

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        # assert (input_size % 32 == 0)

        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks = self._make_bottlenecks()

        # Last convolution has 1280 output channels for scale <= 1
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        self.conv_last = nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2, inplace=True)  # confirmed by paper authors
        self.fc = nn.Linear(self.last_conv_out_ch, self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.c) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.bottlenecks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)

        # average pooling layer
        x = self.avgpool(x)
        x = self.dropout(x)

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) #TODO not needed(?)
# end defining mobileNet_v2 class


class Lane_follow(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial()
        self.omega = 0
        self.omega_list = [0,0,0,0,0]
        self.count = 0
        self.input_img = np.zeros((480,640,3))

        self.alpha = 0.8
        self.speed = 0
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.data_transform = transforms.Compose([transforms.ToTensor()])
        size_downscale = (75, 100)
        self.data_transform = transforms.Compose([transforms.Resize(size_downscale),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # motor omega output
        self.Omega = np.array(
            [0.1, 0.17, 0.24, 0.305, 0.37, 0.44, 0.505, 0.73, -0.1, -0.17, -0.24, -0.305, -0.37, -0.44, -0.505, -0.73,
             0.0, 0.0])
        rospy.loginfo("[%s] Initializing " % (self.node_name))
        self.pub_car_cmd = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)
        self.pub_cam_tilt = rospy.Publisher("/tilt/command", Float64, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb, queue_size=1)
        self.idx = 0

    # load weight
    def initial(self):
        self.model = MobileNet2(input_size=100, num_classes=18)
        checkpoint = torch.load(weight_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

        # self.model = MobileNet2()
        # self.model.load_state_dict(torch.load(weight_path))
        # self.model = self.model.to(self.device)

    def segment(self,img):
        """
        Estimate the robot direction using segmentation approach
        Input:
            img: RGB image (640, 480)
        :return:
        isSegmented = True for image containing robot route
        targetCoor = detegted target point of the robot route
        rotation_rad = robot heading rotation, in radian, counter-clockwise
        """
        isSegmented = False; rotation_rad = 0; targetCoor = ((int)(img.shape[1] / 2), 0); x_vec = 0; y_vec = 0
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

        if (nb_components >= 2):  # at least has one background and segmented foreground
            areas = stats[:, -1]
            max_label = 1  # 0 is for the background, so, skip it
            max_area = areas[1]
            for i in range(2, nb_components):
                if areas[i] > max_area:
                    max_label = i
                    max_area = areas[i]

            center = centroids[max_label].astype(np.int16)
            delta = 20
            up_lim = center[1] + delta;
            low_lim = center[1] - delta
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
                targetCoor = ((int)(centroids[max_label][0]), (int)(center[1]))
                # calculate the rotation degree
                x_vec = (float)((img.shape[1] / 2) - targetCoor[0])
                y_vec = (float)(img.shape[0] - targetCoor[1])
                rotation_rad = (float)(np.arctan(x_vec / y_vec))

        return isSegmented, targetCoor, rotation_rad, (x_vec, y_vec)

    # load image to define omega for motor controlling
    def img_cb(self, data):
        # self.dim = (101, 101)  # (width, height)
        self.count += 1
        self.pub_cam_tilt.publish(1.1)
        if self.count == 2:
            self.count = 0
            try:
                # convert image_msg to cv format
                img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #print("shape:", img.shape)
                # PREDICT using segmentation apprach
                isSegmented, targetCoor, rotation_rad, vec = self.segment(img)
                vec_x, vec_y = vec[0], vec[1]
                img_BGR = cv2.putText(img_BGR, ".", (targetCoor[0], targetCoor[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 2)
                img_BGR = cv2.putText(img_BGR, str(rotation_rad), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                img_BGR = cv2.putText(img_BGR, str(vec_x)+"; " + str(vec_y), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                img_BGR = cv2.line(img_BGR, ((int)(img_BGR.shape[1]/2),(int)(img_BGR.shape[0])), targetCoor, (0,255,0))
                #cv2.imwrite(os.path.join(os.getcwd(),str(self.idx))+".jpg",img_BGR)
                self.idx += 1
                # PREDICT using deep learning approach
                img = ImagePIL.fromarray(np.uint8(img))
                # im1 = img.save(os.path.join(os.getcwd(),str(self.idx)+".jpg"))
                # self.idx += 1
                img = self.data_transform(img)
                #print("img:", img)
                images = torch.unsqueeze(img, 0)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                images = images.to(self.device)
                self.model = self.model.to(self.device)
                output = self.model(images)
                _, top1 = torch.max(output.data, 1)
                # top1 = output.argmax()
                top1 = top1.item()

                # decide using segmentation or deep learning or combined approach
                approach = ['dl', 'improc', 'combined']; approach = approach[1]
                if (approach == 'dl'):
                    self.omega = self.Omega[top1]
                    self.omega_list.append(self.omega)
                    self.omega_list.pop(0)
                elif (approach == 'improc'): # using 'segmentation improc. approach'
                    self.omega = rotation_rad
                    self.omega_list.append(self.omega)
                    self.omega_list.pop(0)
                else: # using combination of the both
                    self.omega = 0.5*self.Omega[top1] + 0.5*rotation_rad
                    self.omega_list.append(self.omega)
                    self.omega_list.pop(0)

                #do smoothing
                #self.omega = (float)(sum(self.omega_list))/(float)(len(self.omega_list))

                # motor control
                car_cmd_msg = Twist()
                cx = 2.5
                if(abs(self.omega)>0.3 and abs(self.omega)<0.5):
                    cx = 1.3
                if(abs(self.omega)>=0.5):
                    cx = 1

                ## speed smoothing
                self.speed = self.alpha * self.speed + (1-self.alpha) * cx
                car_cmd_msg.linear.x = self.speed * 0.123
                car_cmd_msg.angular.z = self.omega * 0.8
                self.pub_car_cmd.publish(car_cmd_msg)

                rospy.loginfo('\nomega: ' + str(self.omega) + '\ntop1: ' + str(top1) + '\nrot_rad:' + str(rotation_rad) + '\n---------')

            except CvBridgeError as e:
                print(e)


if __name__ == "__main__":
    rospy.init_node("lane_follow", anonymous=False)
    lane_follow = Lane_follow()
    rospy.spin()
