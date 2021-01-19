# Line Following Robot
This is the final project for the Deep Learning class lectured by professor Chia-Han Lee in National Chiao Tung University (NCTU), Fall 2020. The objective of the final project is to build a robot to win a line-following competition. This robot has completed the tracks with perfect scores and won first place out of 10 teams. This project used object segmentation techniques for filtering out the line track, and calculates the steering angle and moving speed of the robot. The robot speeds up on straight line tracks and slows down when the curvature of the track is greater than a certain angle.

![lane detection](img/lane_detection.gif)



# Method
This project adopted traditional image processing approach, which is image segmentation via HSV image. There are three general steps: (a) segmenting the line track and calculating the mass center, (b) renewing the *x axis* and (c) calculating the rotation degree. The detailed method is explained in [this video](https://www.youtube.com/watch?v=ocecK87CQw4&list=PLkRkKTC6HZMzyHF8a0tyQuF15H0A4IKVO&ab_channel=ArdianUmam).
![lane detection](img/method.JPG)


# Demonstration
1. Video demo 1, [click here](https://www.youtube.com/watch?v=gqOzMLZzDCs&list=PLkRkKTC6HZMzyHF8a0tyQuF15H0A4IKVO&index=2&ab_channel=ArdianUmam)<br>
2. Video demo 2 (with faster speed), [click here](https://youtu.be/mLA47WiJ1KA)

# Code and Dataset
1. To train and test (infer) the model/algorithm, use `main.py`.
2. To run the robot, use `lane_gogo_from_ori.py`.
3. The dataset can be downloaded [here](https://drive.google.com/drive/folders/1zioCeK1OlrUGLt47aBFgAOjNNyN-MEF6?usp=sharing).


# Team Members
Ardian Umam, [Po-Yuan Jeng](https://github.com/lses40311), [Shih-Wei Chiu](https://github.com/chiu0325), and Jeng-Jung Chen.