import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import os
import sys
import warnings
import myLib


def preprocess_radiograph(img):
    img_median = cv2.medianBlur(img, 5)
    img_bilateral = cv2.bilateralFilter(img_median, 15, 30, 30)
    return scharr_gradient_operator(img_bilateral)


def median_filter(img):
    return cv2.medianBlur(img, 5)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 10, 10)


# def sobel_operator(img):
#     scale = 1
#     delta = 0
#     ddepth = cv2.CV_16S
#
#     grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#     grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#
#     abs_grad_x = cv2.convertScaleAbs(grad_x)
#     abs_grad_y = cv2.convertScaleAbs(grad_y)
#
#     # return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#     return cv2.add(abs_grad_x, abs_grad_y)


def scharr_gradient_operator(img):
    ddepth = cv2.CV_16S

    # Gradient X and Y
    grad_x = cv2.Scharr(img, ddepth, 1, 0).astype(np.float)
    grad_y = cv2.Scharr(img, ddepth, 0, 1).astype(np.float)

    # converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x).astype(np.float)  # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y).astype(np.float)
    x = cv2.add(abs_grad_x, abs_grad_y)

    # normalise it between 0-255
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
    return x_norm.astype(np.uint8)


def read_image(image_name):
    return cv2.imread(image_name, 0)


def cropped_image():
    img = read_image('01.tif')
    # cropped = img[500:1350, 1000:2000]
    return img[700:1100, 1400:1750]


def save_preprocessed_radiograph(img, idx_radiofraph):
    dir_path = "Project_Data/_Data/Radiographs_Preprocessed"
    file_name = "/" + str(idx_radiofraph).zfill(2) + ".tif"
    myLib.ensure_dir(dir_path)
    cv2.imwrite(dir_path + file_name, img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")

    for num_img in range(1, 15):
        path_radiograph = "Project_Data/_Data/Radiographs/" + str(num_img).zfill(2) + ".tif"

        img_org = cv2.imread(path_radiograph, 0)
        img_pro = preprocess_radiograph(img_org)

        # plt.figure()
        # plt.imshow(img_org, cmap='gray', interpolation='bicubic')
        # myLib.move_figure('top-right')
        # plt.title('Original')
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(img_pro, cmap='gray', interpolation='bicubic')
        # myLib.move_figure('bottom-right')
        # plt.title('Original')
        # plt.show()

        # plt.waitforbuttonpress()
        # plt.close('all')

        # save_preprocessed_radiograph(img_pro, num_img)


    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")