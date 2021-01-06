from torchvision import datasets ,models,transforms
import torch
import cv2, os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

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
    # cv2.imshow("mask", mask)

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
                x_vec = (img.shape[1]/2)-targetCoor[0]
                y_vec = img.shape[0]-targetCoor[1]
                rotation_rad = np.arctan(x_vec/y_vec)

        return isSegmented, targetCoor, rotation_rad

    for idx_img in range(100):
        TEST_DIR = "./new_data_2020-1-6"
        list_folder = os.listdir(TEST_DIR); list_folder = sorted(list_folder)
        list_folder = [os.path.join(TEST_DIR, i) for i in list_folder]
        list_img_path = []; class_label = []; list_filename = []
        for idx, i in enumerate(list_folder):
            list_img = os.listdir(i); list_img = sorted(list_img)
            for j in list_img:
                list_img_path.append(os.path.join(i, j))
                list_filename.append(j)

        img = cv2.imread(list_img_path[idx_img])
        # img = cv2.imread("new_data_2020-1-6/01/20.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        isSegmented, targetCoor, degree = segment(img)
        print("Img-"+list_filename[idx_img]+ " contain route: ", isSegmented)
        image = cv2.putText(img, ".", (targetCoor[0], targetCoor[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        image = cv2.putText(image, str(degree), (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        img = cv2.line(image, ((int)(img.shape[1]/2),(int)(img.shape[0])),
                       targetCoor, (0,255,0))
        cv2.imwrite(os.path.join("out_improc", list_filename[idx_img]), img)
        # if(isSegmented):
        #     cv2.imshow(str(i), image)
    cv2.waitKey(0)

    cv2.waitKey()
    cv2.imshow("original", img)
    cv2.waitKey(0)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(hsv_to_rgb(do_square))
    plt.subplot(1, 2, 2)
    plt.figure()
    plt.imshow(hsv_to_rgb(lo_square))
    plt.figure()
    plt.imshow(result)
    # plt.imshow(img)
    plt.show()
