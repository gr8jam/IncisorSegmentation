import cv2
import numpy as np
import os
import sys
import warnings
from matplotlib import pyplot as plt
import matplotlib
import myLib

from IncisorsClass import load_incisors
from ActiveShapeModelClass import ActiveShapeModel
from ImagePreprocessing import preprocess_radiograph
from PointSelectorClass import PointSelector


def outline_incisor(img_radiograph, incisors_idx, init_position):
    # Prepare training set
    num_levels = 3
    incisors = load_incisors([incisors_idx], levels=num_levels)

    # Find the incisors landmarks
    asm = ActiveShapeModel(incisors, img_radiograph, init_position, levels=num_levels, visualization=False)
    landmarks = asm.get_active_shape_model_landmarks()
    return landmarks


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")

    # Import picture to be examined
    file_path = "Project_Data/_Data/Radiographs/extra/20.tif"
    img_radiograph = cv2.imread(file_path, 0)
    img_radiograph = preprocess_radiograph(img_radiograph)

    # Show picture and select initial position of incisor
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    myLib.move_figure(manual_position=[100, 50, 1300, 900])
    plt.imshow(img_radiograph, cmap='gray')
    axes = plt.gca()
    axes.set_xlim([600, 2400])
    axes.set_ylim([1400, 400])
    plt.show()

    for incisor_idx in range(1, 9):
        plt.figure(fig.number)
        message = "\nClick on incisor (idx = " + str(incisor_idx) + ") to estimate initial position..."
        print message
        plt.title(message)

        point_selector = PointSelector(fig)
        init_pos = point_selector.get_point()
        message = "Initial position selected. x = " + str(init_pos[0, 0]) + " , y = " + str(init_pos[1, 0])
        plt.title(message)
        print message

        # Find initial position
        # TODO: Marcel put you algorithm here
        #
        landmarks = outline_incisor(img_radiograph, incisor_idx, init_pos)

        message = "Active shape model algorithm finished."
        plt.title(message)
        print message

        # Show the results
        plt.figure(fig.number)
        # plt.plot(landmarks[0, :], landmarks[1, :], color='r', marker='.', markersize=8)
        plt.plot(landmarks[0, :], landmarks[1, :], color='b', linestyle='-', linewidth=2)

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
