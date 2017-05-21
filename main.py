import cv2
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

import myLib


from IncisorsClass import IncisorShape

if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    # print(os.path.dirname(sys.argv[0]))
    # print(os.path.dirname(os.path.realpath(__file__)))

    print '-----------------------------------'
    print 'Start of the script'

    plt.close('all')
    plt.interactive(False)
    incisors = []

    coord_x = 0
    coord_y = 20

    figCnt = 0
    # Load data from files
    for idx_radiograph in range(5, 8):
        for idx_tooth in range(1, 9):
            # plt.close('all')
            incisors.append(IncisorShape(idx_radiograph, idx_tooth))
            if figCnt < 20:
                coord_x = 5 + coord_x + incisors[-1].showRadiograph(np.array([coord_x, coord_y])) / 1.2
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



