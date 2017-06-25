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


def procrustes_alignment(shapes_list, shape_ref):
    # Superimpose all instances to current reference shape
    for i, shape in enumerate(shapes_list):
        shape.set_ssd(shape_ref.lm_loc)
        shape.roll_lm_for_best_fit(shape_ref.lm_loc)
        shape.set_landmarks_theta(shape_ref.lm_loc)


def procrustes_analysis(shapes_list, visualization=True):
    # Arbitrarily choose a reference shape (typically by selecting it among the available instances)
    # np.random.seed(3)
    # idx = np.random.randint(0, len(shapes_list))
    idx = 9

    # print "idx = " + str(idx)
    shape_ref = ObjectShape(shapes_list[idx].lm_org, shapes_list[idx].img)
    # print "r = " + str(shape_ref.num_image)
    # print "t = " + str(shape_ref.num_tooth)
    # shape_ref.show_radiograph(np.array([800, 200]))

    if visualization:
        shapes_viewer = ShapesViewer(shapes_list, shape_ref)

        shapes_viewer.update_shapes_ref()
        for shape_idx in range(len(shapes_list)):
            shapes_viewer.update_shape_idx(shape_idx)

    # shape_mean = ObjectShape(np.zeros_like(shape_ref.lm_loc))

    iteration_max = 30
    iteration_cnt = 0

    while True:

        if visualization:
            shapes_viewer.update_shapes_all()

        # Superimpose all instances to current reference shape
        procrustes_alignment(shapes_list, shape_ref)

        if visualization:
            shapes_viewer.update_shapes_all()

        # Compute the mean shape of the current set of superimposed shapes
        lm_mean = np.zeros_like(shape_ref.lm_loc, dtype=float)
        for shape in shapes_list:
            lm_mean = lm_mean + shape.lm_loc * shape.scale
        lm_mean = lm_mean / float(len(shapes_list))
        shape_mean = ObjectShape(lm_mean)

        # Compute square distance change of mean shape
        ssdc = np.sum(np.square(shape_ref.lm_org - lm_mean))
        if visualization:
            print "sum of square distance change: " + str(ssdc)

        # Update reference shape
        shape_ref.lm_org = shape_mean.lm_org
        shape_ref.lm_loc = shape_mean.lm_loc
        shape_ref.center = shape_mean.center
        shape_ref.scale = shape_mean.scale
        shape_ref.theta = shape_mean.theta

        if visualization:
            shapes_viewer.update_shapes_ref()

        # End loop if sum of square distance change of mean shape is under certain threshold
        if ssdc < 1e-8:
            if visualization:
                print("Procrustes analysis finished. Square distance change of mean shape was under certain threshold.")
            break

        # End loop if number of iteration exceeds maximum number of allowed iterations
        if iteration_cnt >= iteration_max:
            if visualization:
                print("Procrustes analysis finished. Number of iteration exceeded the maximum allowed iterations.")
            break
        iteration_cnt = iteration_cnt + 1

    # axis_len_0 = 2  # DOF
    # axis_len_1 = len(shapes_list)
    shape_centers_sum = np.zeros((2, 1))
    # shape_theta_sum = 0
    for idx_shape, shape in enumerate(shapes_list):
        shape.set_ssd(shape_ref.lm_loc)
        # print shape.ssd
        shape_centers_sum += shape.center
        # shape_theta_sum += shape.theta

    shape_centers = shape_centers_sum / len(shapes_list)
    # shape_theta = shape_theta_sum / len(shapes_list)
    # print "Avrage theta" + str(shape_theta)
    shape_ref.center = shape_centers
    shape_ref.lm_org += shape_ref.center

    return shape_ref


def separate_good_bad_shape_fit(shapes_list):
    shapes_list_good_fit = []
    shapes_list_bad_fit = []
    for shape in shapes_list:
        if shape.ssd > 1.04:
            shapes_list_bad_fit.append(shape)
        else:
            shapes_list_good_fit.append(shape)
    return shapes_list_good_fit, shapes_list_bad_fit


def print_shapes_fit(shapes_list):
    for shape in shapes_list:
        print shape.ssd


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")

    print("---------------------------")
    print("Start of the script")

    plt.close('all')

    for incisor_idx in range(1, 9):
        myLib.tic()
        # incisor_idx = 7
        incisors = load_incisors([incisor_idx])
        myLib.toc()

        # incisors = incisors[0:5]

        # incisors = create_shapes(6)

        incisor_ref = procrustes_analysis(incisors)

        incisors_good_fit, incisors_bad_fit = separate_good_bad_shape_fit(incisors)

        # shape_viewer_good = ShapesViewer(incisors_good_fit, incisor_ref, "good shapes")
        # shape_viewer_good.update_shapes_ref()
        # shape_viewer_good.update_shapes_all()

        # shape_viewer_bad = ShapesViewer(incisors_bad_fit, incisor_ref, "bad shapes")
        # shape_viewer_bad.update_shapes_ref()
        # shape_viewer_bad.update_shapes_all()

        # myLib.ensure_dir("Project_Data/_Data/Report_Figures")
        # plt.savefig("Project_Data/_Data/Report_Figures/shape_model_" + str(incisor_idx) + ".png")

    print "Bad incisors and their ssd: "
    print_shapes_fit(incisors_bad_fit)

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
