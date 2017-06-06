import numpy as np
from matplotlib import pyplot as plt
import scipy
import os
import sys
import matplotlib

import myLib
import warnings

from ObjectShapeClass import ObjectShape
from ObjectShapeClass import create_shapes
from ShapeViewerClass import ShapesViewer
from IncisorsClass import load_incisors
from ProcrustesAnalysis import procrustes_analysis
from ProcrustesAnalysis import separate_good_bad_shape_fit


def get_profile_intensity_mean(shapes_list):
    procrustes_analysis(shapes_list)

    axis_len_0 = np.size(shapes_list[0].lm_org, axis=1)  # Number of landmarks
    axis_len_1 = 2 * shapes_list[0].k + 1  # Number of samples along profile normal
    axis_len_2 = 5  # Number of levels in gaussian pyramid
    axis_len_3 = len(shapes_list)  # Number of incisors in training set
    profile_intensity_array = np.zeros((axis_len_0, axis_len_1, axis_len_2, axis_len_3))

    for shape_idx, shape in enumerate(shapes_list):
        profile_intensity_array[:, :, :, shape_idx] = shape.profile_intensity

    profile_intensity_mean = np.mean(profile_intensity_array, axis=3)
    return profile_intensity_mean


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

        incisors_profile_intensity_mean = get_profile_intensity_mean(incisors)

        plt.waitforbuttonpress()
        for idx_lm in range(1, 41, 50):
            plt.close("all")

            # Show tooth
            fig_incisor = plt.figure()
            myLib.move_figure('top-right')
            plt.axis('equal')
            level = 4
            incisors[0].show_shape(fig_incisor, level)

            # Highlight the profile of interest
            plt.plot(incisors[0].profile_coordinates[0, idx_lm, :, level],
                     incisors[0].profile_coordinates[1, idx_lm, :, level],
                     color='r', marker='.', markersize=3, linestyle=' ')
            plt.show()

            plt.figure()
            myLib.move_figure('bottom-right')
            plt.grid()
            plt.title("intensity profile, lm = " + str(idx_lm))
            for i in range(np.size(incisors_profile_intensity_mean, axis=2)):
                plt.plot(incisors_profile_intensity_mean[idx_lm, :, i], label="intensity on level " + str(i))
            plt.legend()
            plt.show()
            plt.waitforbuttonpress()

    # shape_viewer = ShapesViewer([incisors[0]], incisors[0], "see profiles")
    # shape_viewer.update_shapes_all()


    incisor_idx = 5
    for level in range(incisors[incisor_idx].levels):
        fig = plt.figure()
        myLib.move_figure("top-left")
        incisors[0].show_shape(fig, level)
        plt.waitforbuttonpress()


    # coord_x = 0
    # coord_y = 20
    # figCnt = 0
    # level = 0
    # for incisor in incisors:
    #     if figCnt < 5:
    #         coord_x += 5 + incisor.show_radiograph(np.array([coord_x, coord_y])) / 1.2
    #         plt.plot(incisor.profile_coordinates[0, :, :, level], incisor.profile_coordinates[1, :, :, level], 'c.', markersize=2)
    #         figCnt = figCnt + 1
    #         # plt.waitforbuttonpress()
    #         if coord_x > 1400:
    #             coord_x = 0
    #             coord_y = coord_y + 300

    # plt.figure()

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
