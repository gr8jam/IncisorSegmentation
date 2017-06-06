import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy
import os
import sys
import matplotlib

import myLib
import warnings


def load_radiograph_image(num_img):
    file_path = "Project_Data/_Data/Radiographs/" + str(num_img).zfill(2) + ".tif"
    img = cv2.imread(file_path, 0)
    print type(img)
    return img


def generate_gaussian_pyramid(img, levels):
    img_lvl = img.copy()
    pyramid = [img_lvl]
    for idx in range(levels - 1):
        img_lvl = cv2.pyrDown(img_lvl, borderType=cv2.BORDER_REPLICATE)
        pyramid.append(img_lvl)
    return pyramid


def show_gaussian_pyramid(pyramid):
    for img in pyramid:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.show()


def save_gaussian_pyramid(pyramid, num_img):
    for level, img in enumerate(pyramid):
        dir_path = "Project_Data/_Data/Gaussian_Pyramid/level_" + str(level)
        file_name = "/radiograph_" + str(num_img).zfill(2) + ".tif"
        myLib.ensure_dir(dir_path)
        cv2.imwrite(dir_path + file_name, img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")

    plt.close('all')

    for idx_img in range(1, 2):
        img_org = load_radiograph_image(idx_img)

        img_pyr = generate_gaussian_pyramid(img_org, 5)

        show_gaussian_pyramid(img_pyr)

        # save_gaussian_pyramid(img_pyr, idx_img)

    print "\nClick to finish process..."
    plt.figure()
    plt.waitforbuttonpress()

    print("==========================")
