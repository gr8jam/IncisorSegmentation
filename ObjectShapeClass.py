import numpy as np
from matplotlib import pyplot as plt
import os
import sys

import myLib
import warnings

from ShapeViewerClass import ShapesViewer


class ObjectShape:
    def __init__(self, lm):
        self.lm_org = lm.astype(float)
        self.center = self.get_landmarks_center()
        self.scale = self.get_landmarks_scale()
        self.theta = 0.0
        self.lm_loc = self.get_landmarks_local()
        self.ssd = -1

    def get_landmarks_center(self):
        k = np.size(self.lm_org, 1)
        x = np.sum(self.lm_org[0, :]) / float(k)
        y = np.sum(self.lm_org[1, :]) / float(k)
        return np.vstack((x, y))

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
        """ Obtain optiaml angle of rotation.
        Help: https://en.wikipedia.org/wiki/Procrustes_analysis """
        a = self.lm_loc[0, :] * lm_ref[1, :] - self.lm_loc[1, :] * lm_ref[0, :]
        b = self.lm_loc[0, :] * lm_ref[0, :] + self.lm_loc[1, :] * lm_ref[1, :]
        numerator = np.sum(a)
        denominator = np.sum(b)
        self.theta = np.arctan2(numerator, denominator)
        self.lm_loc = np.dot(myLib.getRotMatrix(self.theta), self.lm_loc)


def create_shapes(num):
    # lm_org = (np.array([[4,0,0],[0,0,1]])).astype(float)
    lm_org = (np.array([[0, 5, 5, 0], [0, 0, 1, 1]])).astype(float)
    # lm_org = (np.array([[-5, 5, 10, 5, -5, -10], [-5, -5, 0, 5, 5, 0]], dtype=float))  # Hexagon
    # lm_org = (np.array([[-5, -3, 3, 5, 7, 8, 10, 8, 7, 5, 3, -3, -5, -7, -8, -10, -8, -7],
    #                           [-5, -5, -5, -5, -3, -2, 0, 2, 3, 5, 5, 5, 5, 3, 2, 0, -2, -3]], dtype=float))

    obj_shapes = []
    np.random.seed(0)
    angle = np.array([0, 0, 0, 0, 45, -40, -30, -20, 10, 20, 30, 40]) * np.pi / 180.0

    for i in range(num):
        lm_ref = np.copy(lm_org)
        # lm_ref = np.roll(lm_ref, 1, axis=1)
        r, c = lm_ref.shape
        lm = np.copy(lm_ref) + np.random.rand(r, c) * 0.1
        lm = myLib.scalePoints(lm, np.random.rand(1) * 3 - 1)
        lm = myLib.rotatePoints(lm, angle[i])
        lm = myLib.translatePoints(lm, np.random.rand(1) * 10 - 5, np.random.rand(1) * 10 - 5)
        obj_shape = ObjectShape(lm)
        obj_shapes.append(obj_shape)

    return obj_shapes


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")

    plt.close('all')

    # landmarksOrg = (np.array([[4,0,0],[0,0,1]])).astype(float)
    # landmarksOrg = (np.array([[0,1,1,0],[0,0,1,1]])).astype(float)
    landmarksOrg = (np.array([[-5, 5, 10, 5, -5, -10], [-5, -5, 0, 5, 5, 0]], dtype=float))  # Hexagon
    landmarksOrg = (np.array([[-5, -3, 3, 5, 7, 8, 10, 8, 7, 5, 3, -3, -5, -7, -8, -10, -8, -7],
                              [-5, -5, -5, -5, -3, -2, 0, 2, 3, 5, 5, 5, 5, 3, 2, 0, -2, -3]], dtype=float))

    landmarks_ref = np.copy(landmarksOrg)
    shape_ref = ObjectShape(landmarks_ref)

    shapes = create_shapes(3, landmarks_ref)

    shapes_viewer = ShapesViewer(shapes, shape_ref)

    shapes_viewer.update_shapes_ref()
    shapes_viewer.update_shapes_all()

    for i in range(len(shapes)):
        shapes[i].set_landmarks_theta(shape_ref.lm_loc)

    shapes_viewer.update_shapes_all()
    shapes_viewer.update_shapes_ref()

    plt.waitforbuttonpress()
