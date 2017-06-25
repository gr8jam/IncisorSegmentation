from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import Preprocess
import math
import myLib

THRES = 50


def canny_edge_detector(image_name):
    img = Preprocess.cropped_image(image_name)
    return cv2.Canny(img, 10, 15, apertureSize=3)
    # return cv2.Canny(img, 50, 150, apertureSize=3)


def dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def is_near(a, b):
    return dist(a[0:2], b[0:2]) < THRES or dist(a[2:4], b[2:4]) < THRES or \
           dist(a[0:2], b[2:4]) < THRES or dist(a[2:4], b[0:2]) < THRES


def steeper(a, b):
    return a[0] == a[2] or (not b[0] == b[2] and (a[1]-a[3])/(a[0]-a[2]) > (b[1]-b[3])/(b[0]-b[2]))


def is_in_range(line):
    if (line[0] - line[2]) < 3:
        ok = False
        print('im here')
    else:
        ok = 0 < abs((line[1] - line[3]) / (line[0] - line[2])) < 3.7320
    return ok


def filter_lines(h_out):
    r = []
    count1 = 0
    count2 = 0
    print('number of lines originaly : ' + str(h_out.shape[0]))
    for i in range(h_out.shape[0]):
        ok = True
        if is_in_range(h_out[i, 0, :]):
            ok = False
            count1 +=1
        for j in range(i+1, h_out.shape[0]):
            if is_near(h_out[i, 0, :], h_out[j, 0, :]) and steeper(h_out[j, 0, :], h_out[i, 0, :]):
                ok = False
                count2 += 1
                break
        if ok:
            r.append(h_out[i, 0, :])
    print('number of removed lines cause of the angle: ' + str(count1) + ' number of removed lines cause near: '
          + str(count2) + ' number of lines now: ' + str(np.array(r).shape[0]))
    return np.array(r)

def hough_line_transformation(img_name, rho=1, theta=np.pi / 180, threshold=70, min_line_length=100, max_line_gap=80):
    img = Preprocess.cropped_image(img_name)
    edges = canny_edge_detector(img_name)
    # plt.figure()
    # plt.imshow(edges, cmap='gray', interpolation='bicubic')
    # plt.title('Canny edge detection')
    # plt.show()
    lines = filter_lines(cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap))
    # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # for x in range(0, len(lines)):
    #     for x1, y1, x2, y2 in lines[x]:
    #         result = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # without filter
    # a, b, c = lines.shape
    # for i in range(a):
    #     result = cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    # with filter
    a, c = lines.shape
    for i in range(a):
        result = cv2.line(img, (lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]), (0, 0, 255), 3, cv2.LINE_AA)
    return result

if __name__ == '__main__':
    matplotlib.interactive(True)

    image_name = '01.tif'

    # Hough line transformation
    # hough_transformed_image = hough_line_transformation(image_name)

    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    tresh = [30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78]
    plt.figure()
    # for i in list:
    #     for j in tresh:
    #         plt.imshow(hough_line_transformation(str(i) + '.tif', threshold=j), cmap='gray', interpolation='bicubic')
    #         plt.title('hough-' + str(i) + ' treshold ' + str(j))
    #         plt.savefig('hough-' + str(i) + ' treshold ' + str(j) + '.tif')
    #         plt.show()
    edges = canny_edge_detector('10.tif')
    plt.imshow(edges, cmap='gray', interpolation='bicubic')
    plt.title('Canny edge detection')
    plt.savefig('canny_edge_detection.png')
    plt.show()
    plt.waitforbuttonpress()
