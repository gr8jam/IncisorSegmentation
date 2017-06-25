import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates
from gaussian_priamid import generate_gaussian_pyramid
import os
import sys

import myLib
import warnings

from ShapeViewerClass import ShapesViewer


class ObjectShape:
    def __init__(self, lm, img=None, k=5, levels=4):
        self.lm_org = np.copy(lm.astype(float))
        self.center = self.get_landmarks_center(self.lm_org)
        self.scale = self.get_landmarks_scale()
        self.theta = 0.0
        self.lm_loc = self.get_landmarks_local()
        self.ssd = -1
        self.roll = 0
        self.k = k
        self.normals = self.get_normals()
        self.levels = levels
        self.profile_coordinates = self.get_profile_coordinates()
        self.img = np.copy(img)
        if not (img is None):
            self.img = np.copy(img)
            self.img_pyr = generate_gaussian_pyramid(self.img, self.levels)
            self.profile_intensity = self.get_profile_intensity()

    def get_landmarks_center(self, lm):
        # k = np.size(lm, 1)
        # x = np.sum(lm[0, :]) / float(k)
        # y = np.sum(lm[1, :]) / float(k)
        # return np.vstack((x, y))
        center = np.mean(lm, axis=1)
        center = center[:, np.newaxis]
        return center

    def get_landmarks_scale(self):
        d = self.lm_org - self.center
        k = np.size(self.lm_org, 1)
        scale = np.sqrt(np.sum(np.square(d)) / k)
        return scale

    def get_landmarks_local(self):
        lm = (self.lm_org - self.center).astype(float)
        lm = (lm / self.scale.astype(float))
        # lm = np.dot(myLib.getRotMatrix(self.theta), lm)
        return lm

    def set_landmarks_theta(self, lm_ref):
        self.theta = self.get_landmarks_theta(lm_ref)
        self.lm_loc = np.dot(myLib.getRotMatrix(self.theta), self.lm_loc)

    def get_landmarks_theta(self, lm_ref):
        """ Obtain optiaml angle of rotation.
            Help: https://en.wikipedia.org/wiki/Procrustes_analysis """
        a = self.lm_loc[0, :] * lm_ref[1, :] - self.lm_loc[1, :] * lm_ref[0, :]
        b = self.lm_loc[0, :] * lm_ref[0, :] + self.lm_loc[1, :] * lm_ref[1, :]
        numerator = np.sum(a)
        denominator = np.sum(b)
        theta = np.arctan2(numerator, denominator)
        return theta

    def roll_lm_for_best_fit(self, lm_ref):
        self.roll_lm_left(lm_ref)
        self.roll_lm_right(lm_ref)

    def roll_lm(self, lm_ref, shift):
        # Direction determines the roll action
        # shift = -1 : Roll one spot to the left
        # shift = +1 : Roll one spot to the right
        lm_rolled = np.roll(self.lm_loc, shift, axis=1)
        ssd_rolled = self.compute_ssd(lm_ref, lm_rolled)
        if ssd_rolled < self.ssd:
            self.roll = self.roll + shift
            self.ssd = ssd_rolled
            self.roll_shape(shift)
            self.roll_lm(lm_ref, shift)

    def roll_lm_right(self, lm_ref):
        self.roll_lm(lm_ref, +1)

    def roll_lm_left(self, lm_ref):
        self.roll_lm(lm_ref, -1)

    def roll_shape(self, shift):
        self.lm_org = np.roll(self.lm_org, shift, axis=1)
        self.lm_loc = np.roll(self.lm_loc, shift, axis=1)
        self.normals = np.roll(self.normals, shift, axis=0)
        self.profile_coordinates = np.roll(self.profile_coordinates, shift, axis=1)
        self.profile_intensity = np.roll(self.profile_intensity, shift, axis=0)

    def set_ssd(self, lm_ref):
        self.ssd = self.compute_ssd(lm_ref, self.lm_loc)

    def compute_ssd(self, lm_ref, lm_loc):
        return np.sum(np.square(lm_ref - lm_loc)) / len(self.lm_loc)

    def get_normals(self):
        normals = np.zeros(np.size(self.lm_org, axis=1))
        for idx in range(len(normals)):
            lm_prev = self.lm_org[:, (idx - 1) % len(normals)]
            lm_next = self.lm_org[:, (idx + 1) % len(normals)]
            normals[idx] = np.arctan2(lm_next[1] - lm_prev[1], lm_next[0] - lm_prev[0]) + np.pi / 2
        return normals

    def get_profile_coordinates(self):
        axis_len_0 = 2  # Number of DOF
        axis_len_1 = np.size(self.lm_org, axis=1)  # Number of landmarks
        axis_len_2 = 2 * self.k + 1  # Number of samples along profile normal
        axis_len_3 = self.levels  # Number of levels of gaussian pyramid

        profile_coordinates = np.zeros((axis_len_0, axis_len_1, axis_len_2, axis_len_3))

        for level in range(self.levels):
            x0 = self.lm_org[0, :] / (2 ** level) + np.cos(self.normals + np.pi) * self.k
            y0 = self.lm_org[1, :] / (2 ** level) + np.sin(self.normals + np.pi) * self.k
            x1 = self.lm_org[0, :] / (2 ** level) + np.cos(self.normals) * self.k
            y1 = self.lm_org[1, :] / (2 ** level) + np.sin(self.normals) * self.k

            for landmark in range(axis_len_1):
                x = (np.linspace(x0[landmark], x1[landmark], 2 * self.k + 1))
                y = (np.linspace(y0[landmark], y1[landmark], 2 * self.k + 1))
                profile_coordinates[0, landmark, :, level] = x.astype(np.int)
                profile_coordinates[1, landmark, :, level] = y.astype(np.int)
                # profile_coordinates[0, landmark, :, level] = x
                # profile_coordinates[1, landmark, :, level] = y
        return profile_coordinates

    def get_profile_intensity(self):
        axis_len_0 = np.size(self.lm_org, axis=1)  # Number of landmarks
        axis_len_1 = 2 * self.k + 1  # Number of samples along profile normal
        axis_len_2 = self.levels  # Number of levels of gaussian pyramid

        profile_intensity = np.zeros((axis_len_0, axis_len_1, axis_len_2))

        for level in range(axis_len_2):
            for idx in range(axis_len_0):
                x = self.profile_coordinates[0, idx, :, level]
                y = self.profile_coordinates[1, idx, :, level]
                profile_intensity[idx, :, level] = self.img_pyr[level][y.astype(np.int), x.astype(np.int)]  # faster
                # profile_intensity[idx, :, level] = map_coordinates(self.img_pyr[level], np.vstack((y, x)))       # interpolation
                profile_sum = np.sum(profile_intensity[idx, :, level], axis=0)
                if profile_sum == 0:
                    profile_sum = 1
                profile_intensity[idx, :, level] = profile_intensity[idx, :, level] / profile_sum

        return profile_intensity

    def show_shape(self, fig, level=0):
        plt.figure(fig.number)
        plt.axis('equal')
        plt.imshow(self.img_pyr[level], cmap='gray', interpolation='bicubic')

        lm_org_x = self.lm_org[0, :] / (2 ** level)
        lm_org_y = self.lm_org[1, :] / (2 ** level)

        plt.plot(lm_org_x, lm_org_y, color='b', marker='.', markersize=1)  # landmarks
        plt.plot(lm_org_x, lm_org_y, color='b', marker='.', markersize=1)  # start
        plt.plot(lm_org_x, lm_org_y, color='b', linestyle='-', linewidth=1)  # border
        plt.plot(self.profile_coordinates[0, :, :, level], self.profile_coordinates[1, :, :, level],
                 color='c', marker='.', markersize=3, linestyle=' ')  # profile

        window_margin = self.k * 2 ** ((self.levels - 1) - level) * 1.2
        x_max = np.amax(lm_org_x) + window_margin
        x_min = np.amin(lm_org_x) - window_margin
        y_max = np.amax(lm_org_y) + window_margin
        y_min = np.amin(lm_org_y) - window_margin

        axes = plt.gca()
        axes.set_xlim([x_min, x_max])
        axes.set_ylim([y_max, y_min])
        plt.show()


def create_shapes(num):
    # lm_org = (np.array([[4,0,0],[0,0,1]])).astype(float)
    # lm_org = (np.array([[0, 5, 5, 0], [0, 0, 1, 1]])).astype(float)
    lm_org = (np.array([[-5, 5, 10, 5, -5, -10], [-5, -5, 0, 5, 5, 0]], dtype=float))  # Hexagon
    # lm_org = (np.array([[-5, -3, 3, 5, 7, 8, 10, 8, 7, 5, 3, -3, -5, -7, -8, -10, -8, -7],
    #                           [-5, -5, -5, -5, -3, -2, 0, 2, 3, 5, 5, 5, 5, 3, 2, 0, -2, -3]], dtype=float))

    obj_shapes = []
    np.random.seed(0)
    angle = np.array([0, 2, 3, 5, -5, -10, -30, -20, 10, 20, 30, 40]) * np.pi / 180.0

    for i in range(num):
        lm_ref = np.copy(lm_org)
        lm_ref = np.roll(lm_ref, i, axis=1)
        r, c = lm_ref.shape
        lm = np.copy(lm_ref) + np.random.rand(r, c) * 0.1
        lm = myLib.scalePoints(lm, np.random.rand(1) * 3 - 1)
        lm = myLib.rotatePoints(lm, angle[i])
        lm = myLib.translatePoints(lm, np.random.rand(1) * 10 - 5, np.random.rand(1) * 10 - 5) + 150
        # lm_max_x = int(round(np.max(lm[0, :])))
        # lm_max_y = int(round(np.max(lm[1, :])))
        # img = np.ones((lm_max_y + 100, lm_max_x + 100))
        obj_shape = ObjectShape(lm)
        obj_shapes.append(obj_shape)
    return obj_shapes


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")

    plt.close('all')

    shape_ref = create_shapes(1)[0]

    shapes = create_shapes(3)

    shapes_viewer = ShapesViewer(shapes, shape_ref)

    shapes_viewer.update_shapes_ref()
    shapes_viewer.update_shapes_all()

    for i in range(len(shapes)):
        shapes[i].set_landmarks_theta(shape_ref.lm_loc)
        # shapes[i].roll_lm_for_best_fit(shape_ref.lm_loc)

    shapes_viewer.update_shapes_all()
    shapes_viewer.update_shapes_ref()

    plt.waitforbuttonpress()
