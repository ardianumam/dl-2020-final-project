from torchvision import datasets ,models,transforms
import torch
import cv2
from PIL import Image
import numpy as np

MODE = ['loader', 'segmentation']
MODE = MODE[1]

if MODE == 'loader':
    IMG_IN_PATH = "./img_explore/c1/0.1082408_529.jpeg"
    size_downscale = (75, 100)
    transform = transforms.Compose([transforms.ToTensor()])

    #load image from folder and set foldername as label
    train_data = datasets.ImageFolder('img_explore',
                                      transform = transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,shuffle = True)
    for i, data in enumerate(train_loader, 0):
        i += 1
        # get the input
        inputs, labels = data
        inputs2 = inputs * 255
        img_torch = inputs2.detach().numpy()
        # img_torch = img_torch.reshape((480,-1,3))
        img_torch = np.squeeze(img_torch,axis=0)
        img_torch = np.transpose(img_torch, (1, 2, 0))
        img_cv = cv2.imread(IMG_IN_PATH)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.open(IMG_IN_PATH)
        img_pil2 = np.asarray(img_pil)
        cv2.imshow("pil", img_pil2.astype(np.uint8))
        cv2.imshow("torch", img_torch.astype(np.uint8))
        cv2.imshow("cv2", img_cv)
        cv2.waitKey()
        print("dummy...")

if MODE == 'segmentation':
