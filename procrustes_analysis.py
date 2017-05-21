import numpy as np
from matplotlib import pyplot as plt
import os
import sys

import myLib
import warnings

from ObjectShapeClass import ObjectShape
from ShapeViewerClass import ShapesViewer
from IncisorsClass import load_incisors


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")

    print("-----------------------------------")
    print("Start of the script")

    plt.close('all')

    myLib.tic()
    incisors = load_incisors()
    myLib.toc()

    shape_ref = incisors[0]
    incisors = incisors[1:]

    shapes_viewer = ShapesViewer(incisors, shape_ref)

    shapes_viewer.update_shapes_ref()
    shapes_viewer.update_shapes_all()

    plt.waitforbuttonpress()
    for i in range(len(incisors)):
        incisors[i].set_landmarks_theta(shape_ref.lm_loc)

    shapes_viewer.update_shapes_all()
    shapes_viewer.update_shapes_ref()

    print "almost end..."
    plt.waitforbuttonpress()

    a = 10

    print("=====================================")
