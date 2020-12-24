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


class Lane_follow(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial()
        self.omega = 0
        self.count = 0
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_transform = transforms.Compose([transforms.ToTensor()]) 
        # motor omega output
        self.Omega = np.array([0.1,0.17,0.24,0.305,0.37,0.44,0.505,0.73,-0.1,-0.17,-0.24,-0.305,-0.37,-0.44,-0.505,-0.73,0.0,0.0])
        rospy.loginfo("[%s] Initializing " % (self.node_name))
        self.pub_car_cmd = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)
        self.pub_cam_tilt = rospy.Publisher("/tilt/command", Float64, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb, queue_size=1)
    
    # load weight
    def initial(self):
        self.model = CNN_Model()
        self.model.load_state_dict(torch.load(weight_path))
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
                img = self.data_transform(img)
                images = torch.unsqueeze(img,0)
                
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                images = images.to(self.device)
                self.model = self.model.to(self.device)
                output = self.model(images)
                top1 = output.argmax()
                self.omega = self.Omega[top1]
                
                # motor control
                car_cmd_msg = Twist()
                car_cmd_msg.linear.x = 0.123
                car_cmd_msg.angular.z = self.omega*0.64
                rospy.loginfo(str(car_cmd_msg.angular.z))
                self.pub_car_cmd.publish(car_cmd_msg)
                
                rospy.loginfo('\n'+str(self.omega)+'\n'+str(top1))

            except CvBridgeError as e:
                print(e)



if __name__ == "__main__":
    rospy.init_node("lane_follow", anonymous=False)
    lane_follow = Lane_follow()
    rospy.spin()


