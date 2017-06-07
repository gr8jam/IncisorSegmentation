import numpy as np
from matplotlib import pyplot as plt
import cv2
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
from ProcrustesAnalysis import procrustes_alignment
from PCA import create_training_set
from PCA import principal_component_analysis
from profiles import get_profile_intensity_mean
from ImagePreprocessing import preprocess_radiograph


class ShapeModelHandle:
    def __init__(self):
        self.lm_org = plt.plot([], [], color='b', marker='.', markersize=5)[0]
        self.border = plt.plot([], [], color='b', linestyle='-', linewidth=1)[0]
        self.center = plt.plot([], [], color='b', marker='.', markersize=6)[0]

        self.profile = plt.plot([], [], color='c', marker='.', markersize=5, linestyle=' ')[0]
        self.start = plt.plot([], [], color='r', marker='.', markersize=8)[0]


class ShapeTargetHandle:
    def __init__(self):
        self.lm_org = plt.plot([], [], color='g', marker='.', markersize=5)[0]
        self.border = plt.plot([], [], color='g', linestyle='-', linewidth=0.21)[0]
        self.center = plt.plot([], [], color='g', marker='.', markersize=6)[0]
        self.profile = plt.plot([], [], color='g', marker='.', markersize=5)[0]


class ActiveShapeModel:
    def __init__(self, shapes_list, img, init_pos, levels):
        self.init_center = init_pos
        shape_ref = procrustes_analysis(shapes_list, visualization=False)
        eigenval, eigenvec, lm_mu = principal_component_analysis(shapes_list, shape_ref, 3)
        self.eigenvalues = eigenval
        self.eigenvectors = eigenvec
        self.profile_intensity_mean = get_profile_intensity_mean(shapes_list)

        self.current_level = levels - 1

        lm_model = (shape_ref.lm_org - shape_ref.center + self.init_center).astype(np.int)
        self.shape_model = ObjectShape(lm_model, img, k=10, levels=self.current_level + 1)
        self.shape_target = None

        self.b = np.zeros_like(self.eigenvalues)
        self.fig = plt.figure()

        myLib.move_figure('right')
        plt.imshow(self.shape_model.img, cmap='gray', interpolation='bicubic')
        plt.plot(self.init_center[0, 0], self.init_center[1, 0], color='r', marker='.', markersize=5)
        self.handle_model = ShapeModelHandle()
        self.handle_target = ShapeTargetHandle()
        self.update_figure()

        self.update_target_points()
        self.update_figure()
        # plt.waitforbuttonpress()

        self.match_model_to_target()
        self.update_figure()
        # plt.waitforbuttonpress()

    def match_model_to_target(self):

        while True:
            procrustes_alignment([self.shape_model], self.shape_target)

            break

    def update_target_points(self):
        lm_target = np.zeros_like(self.shape_model.lm_org)
        len_k = np.size(self.profile_intensity_mean, axis=1)
        len_ns = np.size(self.shape_model.profile_intensity, axis=1)
        for idx_lm in range(np.size(self.shape_model.lm_org, axis=1)):
            intensity_match = np.zeros((len_ns - len_k + 1,))
            for idx_k in range(len_ns - len_k + 1):
                intensity_error = (self.shape_model.profile_intensity[idx_lm, idx_k:idx_k + len_k, self.current_level] -
                                   self.profile_intensity_mean[idx_lm, :, self.current_level]) ** 2
                intensity_match[idx_k] = np.sum(intensity_error)

            idx_min_error = np.argmin(intensity_match)
            idx_best_match = idx_min_error + (len_k - 1) / 2
            lm_target[:, idx_lm] = self.shape_model.profile_coordinates[:, idx_lm, idx_best_match, self.current_level]

            if (idx_lm > 7) and (idx_lm < 10):
                self.show_profile_intensity_match(idx_lm, intensity_match, idx_min_error)

        self.shape_target = ObjectShape(lm_target * (2**self.current_level))

    def update_figure(self):
        plt.figure(self.fig.number)
        # plt.cla()
        plt.imshow(self.shape_model.img_pyr[self.current_level], cmap='gray', interpolation='bicubic')
        plt.plot(self.init_center[0, 0] / (2 ** self.current_level), self.init_center[1, 0] / (2 ** self.current_level),
                 color='r', marker='.', markersize=5)

        # Update landmarks in coordinate system
        self.handle_model.lm_org.set_xdata(self.shape_model.lm_org[0, :] / (2 ** self.current_level))
        self.handle_model.lm_org.set_ydata(self.shape_model.lm_org[1, :] / (2 ** self.current_level))

        # Update border
        self.handle_model.border.set_xdata(self.shape_model.lm_org[0, :] / (2 ** self.current_level))
        self.handle_model.border.set_ydata(self.shape_model.lm_org[1, :] / (2 ** self.current_level))

        # Update first landmark position
        self.handle_model.start.set_xdata(self.shape_model.lm_org[0, 0] / (2 ** self.current_level))
        self.handle_model.start.set_ydata(self.shape_model.lm_org[1, 0] / (2 ** self.current_level))

        # Update center
        self.handle_model.center.set_xdata(self.shape_model.center[0, :] / (2 ** self.current_level))
        self.handle_model.center.set_ydata(self.shape_model.center[1, :] / (2 ** self.current_level))

        # Update profile coordinates
        self.handle_model.profile.set_xdata(self.shape_model.profile_coordinates[0, :, :, self.current_level])
        self.handle_model.profile.set_ydata(self.shape_model.profile_coordinates[1, :, :, self.current_level])

        if not (self.shape_target is None):
            self.handle_target.lm_org.set_xdata(self.shape_target.lm_org[0, :] / (2 ** self.current_level))
            self.handle_target.lm_org.set_ydata(self.shape_target.lm_org[1, :] / (2 ** self.current_level))

        # recompute the axis limits
        window_margin = 150 / (2 ** self.current_level)
        x_max = self.init_center[0, 0] / (2 ** self.current_level) + window_margin
        x_min = self.init_center[0, 0] / (2 ** self.current_level) - window_margin
        y_max = self.init_center[1, 0] / (2 ** self.current_level) + window_margin
        y_min = self.init_center[1, 0] / (2 ** self.current_level) - window_margin

        axes = plt.gca()
        axes.set_xlim([x_min, x_max])
        axes.set_ylim([y_max, y_min])
        plt.show()

    def show_profile_intensity_match(self, lm_idx, intensity_match, idx_min_error):
        fig_temp = plt.figure()
        plt.subplot(2, 1, 1)

        len_ns = np.size(self.shape_model.profile_intensity, axis=1)
        len_k = np.size(self.profile_intensity_mean, axis=1)
        plt.plot(np.arange(len_ns), self.shape_model.profile_intensity[lm_idx, :, self.current_level], 'b-')
        plt.plot(np.arange(len_k) + idx_min_error, self.profile_intensity_mean[lm_idx, :, self.current_level], 'r-')
        plt.title("idx_lm = " + str(lm_idx))

        plt.subplot(2, 1, 2)
        plt.plot(intensity_match)
        plt.show()
        plt.waitforbuttonpress()
        plt.close(fig_temp.number)


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")
    fig_dummy = plt.figure()

    num_levels = 3
    incisors = load_incisors([5], levels=num_levels)
    file_path = "Project_Data/_Data/Radiographs_Preprocessed/01.tif"
    file_path = "Project_Data/_Data/Segmentations/" + str(1).zfill(2) + "-" + str(5 - 1) + ".png"
    img_radiograph = cv2.imread(file_path, 0)
    # img_radiograph = preprocess_radiograph(img_radiograph)
    pos = np.array([[1400], [1150]])

    asm = ActiveShapeModel(incisors, img_radiograph, pos, levels=num_levels)

    print "\nClick to finish process..."
    plt.figure(fig_dummy.number)
    plt.waitforbuttonpress()

    print("==========================")
