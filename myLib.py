import cv2
# import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt

import os
import sys


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def getRotMatrix(angle):
    ''' angle : float  - angle of rotation in deg '''
    angle = angle * np.pi / 180.0
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
    return rotMatrix


def rotatePoints(points, angle):
    rotMatrix = getRotMatrix(angle)
    rotPoints = np.dot(rotMatrix, np.copy(points))
    return rotPoints


def translatePoints(points, dx, dy):
    transPoints = np.copy(points)
    transPoints[0, :] = transPoints[0, :] + dx
    transPoints[1, :] = transPoints[1, :] + dy
    return transPoints


def scalePoints(points, scale):
    sPoints = np.copy(points) * scale
    return sPoints


def move_figure(position="top-right", manual_position=None):
    """
    Move and resize a window to a set of standard positions on the screen.
    Possible positions are:
    top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
    """
    mgr = plt.get_current_fig_manager()

    if manual_position is None:
        mgr = plt.get_current_fig_manager()
        mgr.full_screen_toggle()  # primitive but works to get screen size
        py = mgr.canvas.height()
        px = mgr.canvas.width()

        d = 10  # width of the window border in pixels
        if position == "top":
            # x-top-left-corner, y-top-left-corner, x-width, y-width (in pixels)
            mgr.window.setGeometry(d, 4 * d, px - 2 * d, py / 2 - 4 * d)
        elif position == "bottom":
            mgr.window.setGeometry(d, py / 2 + 5 * d, px - 2 * d, py / 2 - 4 * d)
        elif position == "left":
            mgr.window.setGeometry(d, 4 * d, px / 2 - 2 * d, py - 4 * d)
        elif position == "right":
            mgr.window.setGeometry(px / 2 + d, 4 * d, px / 2 - 2 * d, py - 4 * d)
        elif position == "top-left":
            mgr.window.setGeometry(d, 4 * d, px / 2 - 2 * d, py / 2 - 4 * d)
        elif position == "top-right":
            mgr.window.setGeometry(px / 2 + d, 4 * d, px / 2 - 2 * d, py / 2 - 4 * d)
        elif position == "bottom-left":
            mgr.window.setGeometry(d, py / 2 + 5 * d, px / 2 - 2 * d, py / 2 - 4 * d)
        elif position == "bottom-right":
            mgr.window.setGeometry(px / 2 + d, py / 2 + 5 * d, px / 2 - 2 * d, py / 2 - 4 * d)
    else:
        mgr.window.setGeometry(manual_position[0], manual_position[1], manual_position[2], manual_position[3])
