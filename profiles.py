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
from IncisorsClass import IncisorShape
from ProcrustesAnalysis import procrustes_analysis
from ProcrustesAnalysis import separate_good_bad_shape_fit

if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")

    plt.close('all')

    myLib.tic()
    incisor = IncisorShape(1, 1)
    myLib.toc()

    plt.figure()
    plt.gca()
    myLib.move_figure('top-right')
    plt.axis('equal')
    plt.grid()
    plt.title("profiles")

    plt.plot(incisor.lm_org[0, :], incisor.lm_org[1, :], 'k.', markersize=5)[0]
    plt.plot(incisor.lm_org[0, :], incisor.lm_org[1, :], 'r--')[0]
    plt.plot(incisor.lm_org[0, 0], incisor.lm_org[1, 0], 'g.', markersize=6)[0]
    plt.plot(incisor.profile_coordinates[0:11, :], incisor.profile_coordinates[11:, :], 'k.', markersize=2)[0]

    plt.figure()

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
