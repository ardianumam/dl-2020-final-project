#!/usr/bin/env python
from __builtin__ import True
import numpy as np
import rospy
import math
import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn
from collections import OrderedDict
import rospkg
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torchvision
import cv2
import os
from torchvision import transforms, utils, datasets
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

os.environ['ROS_IP'] = '10.42.0.2'
bridge = CvBridge()
weight_path = '/home/sis/ncsist_threat_processing/trailnet-testing-Pytorch/2020_summer/src/deep_learning/src/yb_lane.pth'

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
        self.count = 0
        self.last_omega = [0,0,0,0,0]
        self.last_stop = [0,0,0,0,0]
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size_downscale = (75,100)
        self.data_transform = transforms.Compose([transforms.Resize(self.size_downscale),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        # motor omega output
        self.Omega = np.array([0.1,0.17,0.24,0.305,0.37,0.44,0.505,0.73,-0.1,-0.17,-0.24,-0.305,-0.37,-0.44,-0.505,-0.73,0.0,0.0])
        rospy.loginfo("[%s] Initializing " % (self.node_name))
        self.pub_car_cmd = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)
        self.pub_cam_tilt = rospy.Publisher("/tilt/command", Float64, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb, queue_size=1)
    
    # load weight
    def initial(self):
        self.model = MobileNet2(input_size=100, num_classes=19)
        checkpoint = torch.load(weight_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

       
    # load image to define omega for motor controlling
    def img_cb(self, data):
        #self.dim = (101, 101)  # (width, height)
        self.count += 1
        self.pub_cam_tilt.publish(0.9)
        if self.count == 6:
            self.count = 0
            try:
                # convert image_msg to cv format
                img = bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
                #img = cv2.resize(img, self.dim)

                #data_transform = transforms.Compose([
                #    transforms.ToTensor()])
                #change image to PIL
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(np.uint8(img))
                img = self.data_transform(img)
                images = torch.unsqueeze(img,0)

                images = images.to(self.device)
                output = self.model(images)
                top1 = output.argmax()
                self.omega = self.Omega[top1]

                # motor control
                car_cmd_msg = Twist(); temp_ang_z = 0
                if(top1 <= 17): #if the input contains route line
                    car_cmd_msg.linear.x = 0.123
                    temp_ang_z = self.omega*0.64
                    self.last_stop.append(0)
                    self.last_stop.pop()
                else: #if the input contains no route line
                    self.last_stop.append(1)
                    self.last_stop.pop()

                # check if the robot see x blank route, then stop
                isStop = (float)(sum(self.last_stop))/(float)(len(self.last_stop))
                if(isStop == 1):
                    car_cmd_msg.linear.x = 0 #set robot to stop and rotate to find the route
                    # apply heuristic here!

                # perform smoothing for last x omega
                self.last_omega.append(temp_ang_z)
                self.last_omega.pop()
                temp_ang_z = sum(self.last_omega) / (float)(len(self.last_omega))
                car_cmd_msg.angular.z = temp_ang_z

                self.pub_car_cmd.publish(car_cmd_msg)
                rospy.loginfo(str(car_cmd_msg.angular.z))
                rospy.loginfo('\n'+str(self.omega)+'\n'+str(top1))

            except CvBridgeError as e:
                print(e)



if __name__ == "__main__":
    rospy.init_node("lane_follow", anonymous=False)
    lane_follow = Lane_follow()
    rospy.spin()


