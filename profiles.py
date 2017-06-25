import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import matplotlib

import myLib
import warnings

from IncisorsClass import load_incisors
from ProcrustesAnalysis import procrustes_analysis


def get_profile_intensity_mean(shapes_list):
    axis_len_0 = np.size(shapes_list[0].lm_org, axis=1)  # Number of landmarks
    axis_len_1 = 2 * shapes_list[0].k + 1  # Number of samples along profile normal
    axis_len_2 = shapes_list[0].levels  # Number of levels in gaussian pyramid
    axis_len_3 = len(shapes_list)  # Number of incisors in training set
    profile_intensity_array = np.zeros((axis_len_0, axis_len_1, axis_len_2, axis_len_3))

    for shape_idx, shape in enumerate(shapes_list):
        profile_intensity_array[:, :, :, shape_idx] = shape.profile_intensity

    profile_intensity_mean = np.mean(profile_intensity_array, axis=3)
    return profile_intensity_mean


def show_profile_coordinates_along_landmark(shape, idx_lm, level=0):
    fig_shape = plt.figure()
    myLib.move_figure('top-right')
    shape.show_shape(fig_shape, level)

    # Highlight the profile of interest
    plt.plot(shape.profile_coordinates[0, idx_lm, :, level],
             shape.profile_coordinates[1, idx_lm, :, level],
             color='r', marker='.', markersize=3, linestyle=' ')
    plt.show()


def show_profile_intensity_mean_along_landmark(profile_intensity_mean, idx_lm):
    plt.figure()
    myLib.move_figure('bottom-right')
    plt.grid()
    plt.title("intensity profile, lm = " + str(idx_lm))
    for i in range(np.size(profile_intensity_mean, axis=2)):
        plt.plot(profile_intensity_mean[idx_lm, :, i], label="intensity on level " + str(i))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")

    plt.close('all')

    incisor_idx_list = np.arange(1, 9)
    incisor_idx_list = [5]

    for incisor_idx in incisor_idx_list:
        myLib.tic()
        # incisor_idx = 8
        incisors = load_incisors([incisor_idx])
        # incisors = load_incisors([5, 6, 7, 8])
        myLib.toc()

        procrustes_analysis(incisors)
        incisors_profile_intensity_mean = get_profile_intensity_mean(incisors)

        incisor_idx = 5

        print "\nIntensity profile along boundary normals."
        for idx_landmark in range(0, 40, 3):
            plt.close("all")
            show_profile_coordinates_along_landmark(incisors[incisor_idx], idx_landmark, level=2)
            show_profile_intensity_mean_along_landmark(incisors_profile_intensity_mean, idx_landmark)
            print "   Results for landmark number = " + str(idx_landmark) + ". Press button to continue..."
            plt.waitforbuttonpress()

        # shape_viewer = ShapesViewer([incisors[0]], incisors[0], "see profiles")
        # shape_viewer.update_shapes_all()

        print "\nBoundary normals on different levels."
        for level in range(incisors[incisor_idx].levels):
            fig = plt.figure()
            myLib.move_figure("top-left")
            incisors[incisor_idx].show_shape(fig, level)
            print "   Level = " + str(level) + ". Press button to continue..."
            plt.waitforbuttonpress()

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
