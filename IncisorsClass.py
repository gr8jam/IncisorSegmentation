import numpy as np
import cv2
from matplotlib import pyplot as plt

import os
import sys
import warnings

import myLib
from ObjectShapeClass import ObjectShape


class IncisorShape(ObjectShape):
    """" Class represents incisor shape"""

    def __init__(self, num_img, num_tth):
        self.num_image = num_img
        self.num_tooth = num_tth

        self.pathLandmarks = "Project Data/_Data/Landmarks/original/landmarks" + str(num_img) + "-" + str(
            num_tth) + ".txt"

        points = np.loadtxt(self.pathLandmarks)
        landmarks = np.zeros((2, 40))
        landmarks[0, :] = points[::2]
        landmarks[1, :] = points[1::2]

        ObjectShape.__init__(self, landmarks)
        del points
        del landmarks

        # self.path_radiograph = "Project Data/_Data/Radiographs/" + str(num_img).zfill(2) + ".tif"
        # self.radiograph = np.zeros((1, 1))
        # self.radiograph = cv2.imread(self.path_radiograph, 0)
        #
        # self.path_segmentation = "Project Data/_Data/Segmentations/" + str(num_img).zfill(2) + "-" + str(
        #     num_tth - 1) + ".png"
        # self.segmentation = np.zeros((1, 1))
        # self.segmentation = cv2.imread(self.path_segmentation, 0)

    def show_radiograph(self, position):
        plt.figure()
        plt.imshow(self.radiograph, cmap='gray', interpolation='bicubic')
        x = self.lm_org[0, :]
        y = self.lm_org[1, :]
        maxx = np.amax(x) + 10
        minx = np.amin(x) - 10
        maxy = np.amax(y) + 10
        miny = np.amin(y) - 10
        plt.plot(x, y, 'r.', markersize=10)
        plt.plot(x[0], y[0], 'w.', markersize=10)
        plt.plot(x[1], y[1], 'b.', markersize=10)
        plt.plot(x[20], y[20], 'm.', markersize=10)
        plt.plot(x[21], y[21], 'b.', markersize=10)
        axes = plt.gca()
        axes.set_xlim([minx, maxx])
        axes.set_ylim([maxy, miny])
        myLib.move_figure('', np.hstack((position, np.array([2 * (maxx - minx), maxy - miny]))))
        plt.show(block=False)
        return 2 * (maxx - minx)

    def show_segmentation(self):
        plt.figure()
        plt.imshow(self.segmentation, cmap='gray', interpolation='bicubic')
        myLib.move_figure('bottom-right')
        x = self.lm_org[0, ::5]
        y = self.lm_org[1, ::5]
        plt.plot(x, y, 'r.', linewidth=1.0)
        plt.show()


def load_incisors():
    incisors_list = []
    for idx_radiograph in range(1, 14):  # 1 - 14
        for idx_tooth in range(1, 5):   # 1 - 8
            # plt.close('all')
            incisors_list.append(IncisorShape(idx_radiograph, idx_tooth))
    return incisors_list


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")

    print("-----------------------------------")
    print("Start of the script")

    plt.close('all')
    plt.interactive(False)
    incisors = []

    coord_x = 0
    coord_y = 20

    figCnt = 0
    # Load data from files
    for idx_radiograph in range(7, 8):
        for idx_tooth in range(1, 9):
            # plt.close('all')
            incisors.append(IncisorShape(idx_radiograph, idx_tooth))
            if figCnt < 20:
                coord_x = 5 + coord_x + incisors[-1].show_radiograph(np.array([coord_x, coord_y])) / 1.2
                figCnt = figCnt + 1
                # plt.waitforbuttonpress()
                if coord_x > 1400:
                    coord_x = 0
                    coord_y = coord_y + 300

                    # incisors[-1].showSegmentation()

                    # print incisors[-1].landmarks[0,:]
                    # print incisors[-1].landmarks[1,:]

                    # plt.waitforbuttonpress()
    plt.waitforbuttonpress()
    print("=====================================")
