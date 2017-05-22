import numpy as np
from matplotlib import pyplot as plt
import os
import sys

import myLib
import warnings

from ObjectShapeClass import ObjectShape
from ObjectShapeClass import create_shapes
from ShapeViewerClass import ShapesViewer
from IncisorsClass import load_incisors


def procrustes_analysis(shapes_list):

    # Arbitrarily choose a reference shape (typically by selecting it among the available instances)
    np.random.seed(5)
    idx = np.random.randint(0, len(shapes_list))
    idx = 0
    shape_ref = shapes_list[idx]

    shapes_viewer = ShapesViewer(shapes_list, shape_ref)

    # shape_mean = ObjectShape(np.zeros_like(shape_ref.lm_loc))
    lm_mean = np.zeros_like(shape_ref.lm_loc)
    lm_mean_old = lm_mean

    iteration_max = 30
    iteration_cnt = 0

    while (True):
        # Reset the the mean estimate of landmarks
        lm_mean = lm_mean * 0

        # Superimpose all instances to current reference shape
        for shape_idx in range(len(shapes_list)):
            shapes_list[shape_idx].set_landmarks_theta(shape_ref.lm_loc)
            lm_mean = np.dstack((lm_mean, shapes_list[shape_idx].lm_loc))
            shapes_viewer.update_shape_idx(shape_idx)

        shapes_viewer.update_shapes_ref()
        # plt.waitforbuttonpress()

        # Compute the mean shape of the current set of superimposed shapes
        lm_mean = lm_mean[:, :, 1:]
        lm_mean = np.mean(lm_mean, axis=2)
        shape_mean = ObjectShape(lm_mean)

        # Compute square distance change of mean shape
        ssdc = np.sum(np.square(shape_ref.lm_org - lm_mean))
        print "sum of square distance change: " + str(ssdc)

        # Update reference shape
        shape_ref.lm_org = shape_mean.lm_org
        shape_ref.lm_loc = shape_mean.lm_loc
        shape_ref.center = shape_mean.center
        shape_ref.scale = shape_mean.scale
        shape_ref.theta = shape_mean.theta

        # End loop if sum of square distance change of mean shape is under certain threshold
        if ssdc < 1e-15:
            print("Procrustes analysis finished. Square distance change of mean shape was under certain threshold.")
            break

        # End loop if number of iteration exceeds maximum number of allowed iterations
        if iteration_cnt >= iteration_max:
            print("Procrustes analysis finished. Number of iteration exceeded the maximum allowed iterations.")
            break
        iteration_cnt = iteration_cnt + 1



if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")

    print("---------------------------")
    print("Start of the script")

    plt.close('all')

    myLib.tic()
    incisors = load_incisors()
    myLib.toc()

    # shape_ref = incisors[0]
    # incisors = incisors[0:5]

    # shapes_viewer = ShapesViewer(incisors, shape_ref)

    # landmarksOrg = (np.array([[4,0,0],[0,0,1]])).astype(float)
    # landmarksOrg = (np.array([[0,1,1,0],[0,0,1,1]])).astype(float)
    # landmarksOrg = (np.array([[-5, 5, 10, 5, -5, -10], [-5, -5, 0, 5, 5, 0]], dtype=float))  # Hexagon
    # landmarksOrg = (np.array([[-5, -3, 3, 5, 7, 8, 10, 8, 7, 5, 3, -3, -5, -7, -8, -10, -8, -7],
    #                           [-5, -5, -5, -5, -3, -2, 0, 2, 3, 5, 5, 5, 5, 3, 2, 0, -2, -3]], dtype=float))

    # landmarks_ref = np.copy(landmarksOrg)
    # shapes = create_shapes(2, landmarks_ref)

    procrustes_analysis(incisors)
    # shapes_viewer.update_shapes_ref()
    # shapes_viewer.update_shapes_all()
    #
    # plt.waitforbuttonpress()
    # for i in range(len(incisors)):
    #     incisors[i].set_landmarks_theta(shape_ref.lm_loc)
    #
    # shapes_viewer.update_shapes_all()
    # shapes_viewer.update_shapes_ref()

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
