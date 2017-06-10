import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import sys
import matplotlib

import myLib
import warnings

from ObjectShapeClass import ObjectShape
from IncisorsClass import load_incisors
from ProcrustesAnalysis import procrustes_analysis
from ProcrustesAnalysis import procrustes_alignment
from PCA import principal_component_analysis
from PCA import reconstruct_shape_object
from PCA import project_shape_to_principal_components_space
from profiles import get_profile_intensity_mean
from ImagePreprocessing import preprocess_radiograph


class ActiveShapeModel:
    def __init__(self, shapes_list, img, init_pos, levels):
        self.init_center = np.copy(init_pos)
        self.img = np.copy(img)

        self.shape_ref = procrustes_analysis(shapes_list, visualization=False)
        eigenval, eigenvec, lm_mu = principal_component_analysis(shapes_list, self.shape_ref, 5)
        self.eigenvalues = eigenval
        self.eigenvectors = eigenvec
        self.profile_intensity_mean = get_profile_intensity_mean(shapes_list)

        self.current_level = levels - 1

        # lm_model = (self.shape_ref.lm_org - self.shape_ref.center + self.init_center).astype(np.int)
        lm_model = (self.shape_ref.lm_org - self.shape_ref.center) / 1.30
        lm_model = (lm_model + self.init_center).astype(np.int)
        self.shape_model = ObjectShape(lm_model, self.img, k=6, levels=self.current_level + 1)
        self.shape_target = None

        self.b = np.zeros_like(self.eigenvalues)
        self.fig = plt.figure()

        myLib.move_figure('right')
        plt.imshow(self.shape_model.img, cmap='gray', interpolation='bicubic')
        plt.plot(self.init_center[0, 0], self.init_center[1, 0], color='r', marker='.', markersize=5)
        self.update_figure()

        # self.update_target_points()
        # self.update_figure()
        # plt.waitforbuttonpress()
        #
        # self.match_model_to_target()
        # self.update_figure()
        # plt.waitforbuttonpress()

        # plt.waitforbuttonpress()
        # self.active_shape_model_algorithm()

        plt.waitforbuttonpress()
        self.multi_resolution_search()
        self.update_target_points()

    def multi_resolution_search(self):
        while self.current_level >= 0:
            print self.current_level
            self.active_shape_model_algorithm()
            self.current_level -= 1
            plt.waitforbuttonpress()
        self.current_level = 0

    def active_shape_model_algorithm(self):
        for i in range(5):
            self.update_target_points()
            self.match_model_to_target()
            self.update_figure()
            # plt.waitforbuttonpress()

    def match_model_to_target(self):

        b = self.b * 0
        b_old = np.copy(b)

        max_iter = 3
        num_iter = 0

        while True:

            b = project_shape_to_principal_components_space(self.shape_target, self.shape_ref, self.eigenvectors)

            limit = 2
            for idx, param in enumerate(b):
                if b[idx] > limit * np.sqrt(self.eigenvalues[idx]):
                    b[idx] = limit * np.sqrt(self.eigenvalues[idx])
                elif b[idx] < -limit * np.sqrt(self.eigenvalues[idx]):
                    b[idx] = -limit * np.sqrt(self.eigenvalues[idx])

            shape_new_model = reconstruct_shape_object(self.shape_ref, self.eigenvectors, b)
            theta = shape_new_model.get_landmarks_theta(self.shape_target.lm_loc)
            lm_model = np.dot(myLib.getRotMatrix(theta), shape_new_model.lm_loc)
            lm_model = lm_model * self.shape_target.scale + self.shape_target.center
            self.shape_model = ObjectShape(lm_model, self.img, k=8, levels=self.current_level + 1)

            procrustes_alignment([self.shape_model], self.shape_target)
            lm_new = self.shape_model.lm_loc * self.shape_target.scale + self.shape_target.center
            self.shape_model = ObjectShape(lm_new, self.img, k=8, levels=self.current_level + 1)

            b_change = b - b_old
            b_old = np.copy(b)

            if np.sum(b_change) < 1e-10:
                break

            num_iter += 1
            if num_iter > max_iter:
                break

    def update_target_points(self):
        lm_target = np.zeros_like(self.shape_model.lm_org)
        len_k = np.size(self.profile_intensity_mean, axis=1)
        len_ns = np.size(self.shape_model.profile_intensity, axis=1)
        for idx_lm in range(np.size(self.shape_model.lm_org, axis=1)):
            intensity_match = np.zeros((len_ns - len_k + 1,))
            intensity_similarity = np.zeros((len_ns - len_k + 1,))
            for idx_k in range(len_ns - len_k + 1):
                model_intensity = self.shape_model.profile_intensity[idx_lm, idx_k:idx_k + len_k, self.current_level]
                mean_intensity = self.profile_intensity_mean[idx_lm, :, self.current_level]
                intensity_error = (model_intensity - mean_intensity)
                intensity_error = intensity_error ** 2
                intensity_match[idx_k] = np.sum(intensity_error)

                inner_product = model_intensity * mean_intensity

                intensity_similarity[idx_k] = np.sum(inner_product)


                # if (idx_lm > 0) and (idx_lm < 2):
                #     plt.figure()
                #     plt.title('shift = ' + str(idx_k))
                #     plt.plot(model_intensity, 'b-')
                #     plt.plot(mean_intensity, 'r-')
                #     plt.show()
                #     plt.waitforbuttonpress()
                #     # plt.close()

            idx_min_error = np.argmin(intensity_match)
            idx_min_error = np.argmax(intensity_similarity)

            idx_best_match = idx_min_error + (len_k - 1) / 2

            lm_target[:, idx_lm] = self.shape_model.profile_coordinates[:, idx_lm, idx_best_match, self.current_level]

            # if (idx_lm > 0) and (idx_lm < 2):  # and (self.current_level == 1):
            #     # self.show_profile_intensity_match(idx_lm, intensity_match, idx_min_error)
            #     self.show_profile_intensity_match(idx_lm, intensity_similarity, idx_min_error)

        self.shape_target = ObjectShape(lm_target * (2 ** self.current_level))
        self.update_figure()

    def update_figure(self):
        plt.figure(self.fig.number)
        plt.cla()
        plt.title("Level = " + str(self.current_level))

        # Show image at current level
        plt.imshow(self.shape_model.img_pyr[self.current_level], cmap='gray', interpolation='bicubic')

        # Plot initial position
        plt.plot(self.init_center[0, 0] / (2 ** self.current_level),
                 self.init_center[1, 0] / (2 ** self.current_level),
                 color='r', marker='.', markersize=8)

        # Update model's profile coordinates
        plt.plot(self.shape_model.profile_coordinates[0, :, :, self.current_level],
                 self.shape_model.profile_coordinates[1, :, :, self.current_level],
                 color='c', marker='.', markersize=5, linestyle=' ')

        # Draw model's center
        plt.plot(self.shape_model.center[0, 0] / (2 ** self.current_level),
                 self.shape_model.center[1, 0] / (2 ** self.current_level),
                 color='b', marker='.', markersize=5)

        # Draw model's landmarks
        plt.plot(self.shape_model.lm_org[0, :] / (2 ** self.current_level),
                 self.shape_model.lm_org[1, :] / (2 ** self.current_level),
                 color='b', marker='.', markersize=5)

        # Draw model's first landmark
        plt.plot(self.shape_model.lm_org[0, 12] / (2 ** self.current_level),
                 self.shape_model.lm_org[1, 12] / (2 ** self.current_level),
                 color='m', marker='.', markersize=8)

        # Draw model's border
        plt.plot(self.shape_model.lm_org[0, :] / (2 ** self.current_level),
                 self.shape_model.lm_org[1, :] / (2 ** self.current_level),
                 color='b', linestyle='-', linewidth=1)

        if not (self.shape_target is None):
            # Draw target's landmarks
            plt.plot(self.shape_target.lm_org[0, :] / (2 ** self.current_level),
                     self.shape_target.lm_org[1, :] / (2 ** self.current_level),
                     color='g', marker='.', markersize=5)

            # Draw target's first landmark
            plt.plot(self.shape_target.lm_org[0, 0] / (2 ** self.current_level),
                     self.shape_target.lm_org[1, 0] / (2 ** self.current_level),
                     color='r', marker='.', markersize=8)

            # Draw target's border
            plt.plot(self.shape_target.lm_org[0, :] / (2 ** self.current_level),
                     self.shape_target.lm_org[1, :] / (2 ** self.current_level),
                     color='g', linestyle='-', linewidth=1)

        # recompute the axis limits
        window_margin = 350 / (2 ** self.current_level)
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
        plt.plot([idx_min_error, idx_min_error],
                 [0, np.max([np.max(self.profile_intensity_mean[lm_idx, :, self.current_level]),
                             np.max(self.shape_model.profile_intensity[lm_idx, :, self.current_level])])], 'r--')
        plt.plot([idx_min_error + len_k - 1, idx_min_error + len_k - 1],
                 [0, np.max([np.max(self.profile_intensity_mean[lm_idx, :, self.current_level]),
                             np.max(self.shape_model.profile_intensity[lm_idx, :, self.current_level])])], 'r--')

        plt.title("idx_lm = " + str(lm_idx))

        plt.subplot(2, 1, 2)
        plt.plot(intensity_match)
        plt.show()
        plt.waitforbuttonpress()
        plt.close(fig_temp.number)

    def rigid_aligment(self, shape, shape_ref):
        shape.set_landmarks_theta(shape_ref.lm_loc)
        lm_new = shape.lm_loc
        lm_new = lm_new + shape_ref.scale
        lm_new = lm_new + shape_ref.center
        return lm_new


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
    # file_path = "Project_Data/_Data/Radiographs/08.tif"
    # file_path = "Project_Data/_Data/Segmentations_Preprocessed/02.tif"
    img_radiograph = cv2.imread(file_path, 0)
    # img_radiograph = preprocess_radiograph(img_radiograph)
    pos = np.array([[1410], [1125]])

    asm = ActiveShapeModel(incisors, img_radiograph, pos, levels=num_levels)

    print "\nClick to finish process..."
    plt.figure(fig_dummy.number)
    plt.waitforbuttonpress()

    print("==========================")
