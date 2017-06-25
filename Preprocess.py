import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import warnings
from numpy.distutils.system_info import x11_info
import myLib



def preprocess_radiograph(img):
    img = cv2.medianBlur(img, 5)
    img = cv2.bilateralFilter(img, 15, 30, 30)
    return scharr_gradient_operator(img)


def read_image(image_name):
    return cv2.imread(image_name, 0)


def cropped_image(image_name):
    img = bilateral_filter(median_filter(read_image(image_name)))
    return img[450:1250, 1250:1800]
    #return img[500:1350, 1000:2000]
    # return img[700:1100, 1400:1750]


def median_filter(img):
    return cv2.medianBlur(img, 5)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 10, 10)


def sobel_operator(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return cv2.add(abs_grad_x, abs_grad_y)


def plot_radiograph_image(image_name, title, position):
    plt.figure()
    plt.imshow(cropped_image(image_name), cmap='gray', interpolation='bicubic')
    plt.title(title)
    myLib.move_figure(position)
    plt.waitforbuttonpress()
    plt.show()


def plot_test_image(image_name, title, position):
    plt.figure()
    plt.imshow(read_image(image_name), cmap='gray', interpolation='bicubic')
    plt.title(title)
    myLib.move_figure(position)
    plt.waitforbuttonpress()
    plt.show()


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

if __name__ == '__main__':
    matplotlib.interactive(True)
    sigma = 30
    name = '05.tif'
    # plot_test_image('test2.tif', 'test2 picture', 'top-right')

    plt.figure()
    plt.imshow(read_image('01.tif'), cmap='gray', interpolation='bicubic')
    plt.title('Original image')
    plt.show()
    plt.savefig('original.png')

    plt.imshow(median_filter(cropped_image('01.tif')), cmap='gray', interpolation='bicubic')
    plt.title('Median filtered image')
    plt.show()
    plt.savefig('median.png')

    plt.imshow(bilateral_filter(median_filter(cropped_image('01.tif'))), cmap='gray', interpolation='bicubic')
    plt.title('Bilateral filtered image')
    plt.show()
    plt.savefig('bilateral.png')

    plt.imshow(scharr_gradient_operator(bilateral_filter(median_filter(cropped_image('01.tif')))), cmap='gray', interpolation='bicubic')
    plt.title('Scharr gradient operator image')
    plt.show()
    plt.savefig('scharr.png')
    plt.waitforbuttonpress()
