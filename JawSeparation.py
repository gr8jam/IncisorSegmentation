import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import warnings

import Preprocess

JAW__TOP_THRESHOLD = 100
JAW_BOTTOM_THRESHOLD = 20

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def sum_row_pixel_intensity(image, row):
    total = 0
    # print(image.shape)
    for column in range(image.shape[1]):
        total += image[row, column]
        # print("row: " + str(row) + ' column: ' + str(column) + ' intensity: '
        # + str(image[row, column]) + ' total: ' + str(total))
    # print(str(total) + ' is the sum of the intensity at line ' + str(row))
    return total


def local_min_upper_jaw(image_name, line):
    image = Preprocess.cropped_image(image_name)
    min_row_value = 100000
    for row in range(line-JAW__TOP_THRESHOLD, line-JAW_BOTTOM_THRESHOLD):
        #print(row)
        n = sum_row_pixel_intensity(image, row)
        if(n < min_row_value) and (n != 0):
            min_row_value = n
            line = row
    print('UPPER JAW LINE: ' + str(line))
    return line


def local_min_lower_jaw(image_name, line):
    image = Preprocess.cropped_image(image_name)
    min_row_value = 100000
    for row in range(line+JAW_BOTTOM_THRESHOLD, line+JAW__TOP_THRESHOLD):
        n = sum_row_pixel_intensity(image, row)
        if(n < min_row_value) and (n != 0):
            min_row_value = n
            line = row
    print('LOWER JAW LINE: ' + str(line))
    return line


def global_min_row_intensity(image_name):
    image = Preprocess.cropped_image(image_name)
    line = 0
    min_row_value = 100000
    # n = sum_row_pixel_intensity(image, 0)
    for row in range(image.shape[0]):
        # n = sum_column_pixel_intensity(img.shape[0], column)
        n = sum_row_pixel_intensity(image, row)
        if(n < min_row_value) and (n != 0):
            min_row_value = n
            line = row
    print('GLOBAL JAW LINE: ' + str(line))
    return line


def plot_the_jaws_separate_line(row, image_name):
    image = Preprocess.cropped_image(image_name)
    plt.figure(1)
    plt.imshow(cv2.line(image, (0, row), (img.shape[1], row), (255, 0, 0), 2), cmap='gray', interpolation='bicubic')
    plt.title('Jaw separation-'+ image_name)
    plt.savefig('jaw_separation-' + image_name)
    plt.show()

if __name__ == '__main__':
    # read the preprocessed image
    img = Preprocess.cropped_image('02.tif')
    #list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # calculate the global minimum intensity of the rows - we use the gray image to compute the pixel intensity
    list = [2, 2]
    plt.figure(1)
    for i in list:
        image_name = str(i) + '.tif'
        global_min_row = global_min_row_intensity(image_name)
        #upper_jaw = local_min_upper_jaw(image_name, global_min_row)
        #lower_jaw = local_min_lower_jaw(image_name, global_min_row)

        #lines = [global_min_row, upper_jaw, lower_jaw]
        lines = [global_min_row, global_min_row]
        image = Preprocess.cropped_image(image_name)
        print(image.shape)
        for j in lines:
            result = cv2.line(image, (0, j), (image.shape[1], j), (255, 0, 0), 2)



        plt.imshow(result, cmap='gray', interpolation='bicubic')
        plt.title('Jaw separation')
        plt.savefig('jaw_separation_toproject-' + image_name)
        plt.show()
    plt.waitforbuttonpress()
        # # plot the result of the jaws separation
        # plot_the_jaws_separate_line(global_min_row, str(j) + '.tif')
        # plot_the_jaws_separate_line(upper_jaw, str(j) + '.tif')
        # plot_the_jaws_separate_line(lower_jaw, str(j) + '.tif')
