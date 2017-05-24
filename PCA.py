import numpy as np
from matplotlib import pyplot as plt
import scipy
import os
import sys

import myLib
import warnings

# from ObjectShapeClass import ObjectShape
# from ObjectShapeClass import create_shapes
from ShapeViewerClass import ShapesViewer
from IncisorsClass import load_incisors
from ProcrustesAnalysis import procrustes_analysis

from scipy.sparse.linalg import eigs

def principal_component_analysis(X, num_pc):
    """
        Do a PCA analysis on X.

        Args:
            X (np.array):   Training set with vertically stacked training vectors xi, with dimensions [d x n], 
                            where [d] is the number of points (dimensions) in training vector and [n] is number of 
                            training vectors.
            num_pc (int):   Number of principal components that are of user's interest

        Returns:
            eigen_values (np.array) : num_pc largest eigenvalues of the covariance matrix
            eigen_vectors (np.array): num_pc largest eigenvectors of the covariance matrix corresponding to eigenvalues
            mu (np.array)           : Average sample.

    """
    [d, n] = X.shape
    if (num_pc <= 0) or (num_pc > n):
        num_pc = n

    # Compute average
    mu = np.mean(X, axis=1)

    # Compute correlation matrix C
    C = np.zeros((d, d))
    for i in range(n):
        C = C + np.dot((X[:, i] - mu).reshape(d, 1), (X[:, i] - mu).reshape(1, d))
    C = C / (n - 1)

    # Compute eigenvalues and eigenvectors
    # eigen_values, eigen_vectors = scipy.sparse.linalg.eigs(C, k=num_pc, which='LM')
    eigen_values, eigen_vectors = eigs(C, k=num_pc, which='LM')
    eigen_values = np.real(eigen_values)
    eigen_vectors = np.real(eigen_vectors)

    return eigen_values, eigen_vectors, mu


def create_training_set(shapes_list):
    """
        Create an training set with horizontally stacking landmarks from all the shapes in shapes_list     
        :param shapes_list: 
        :return: 
    """
    n = len(shapes_list)
    d = len(shapes_list[0].lm_loc.flatten())
    X = np.zeros((d, n), dtype=float)

    for i, shape in enumerate(shapes_list):
        X[:, i] = shape.lm_loc.flatten()

    return X


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")

    print("---------------------------")
    print("Start of the script")

    plt.close('all')

    myLib.tic()
    incisors = load_incisors()
    myLib.toc()

    # incisors = incisors[0:5]

    # landmarksOrg = (np.array([[4,0,0],[0,0,1]])).astype(float)
    # landmarksOrg = (np.array([[0, 5, 5, 0], [0, 0, 1, 1]])).astype(float)
    # landmarksOrg = (np.array([[-5, 5, 10, 5, -5, -10], [-5, -5, 0, 5, 5, 0]], dtype=float))  # Hexagon
    # landmarksOrg = (np.array([[-5, -3, 3, 5, 7, 8, 10, 8, 7, 5, 3, -3, -5, -7, -8, -10, -8, -7],
    #                           [-5, -5, -5, -5, -3, -2, 0, 2, 3, 5, 5, 5, 5, 3, 2, 0, -2, -3]], dtype=float))
    #
    # landmarks_ref = np.copy(landmarksOrg)
    # incisors = create_shapes(3, landmarks_ref)

    incisor_ref = procrustes_analysis(incisors)

    incisors_bad_lm = []
    incisors_good_lm = []
    for incisor in incisors:
        if incisor.ssd > 0.04:
            incisors_bad_lm.append(incisor)
        else:
            incisors_good_lm.append(incisor)

    shape_viewer_good = ShapesViewer(incisors_good_lm, incisor_ref)
    shape_viewer_good.update_shapes_ref()
    shape_viewer_good.update_shapes_all()

    # shape_viewer_bad = ShapesViewer(incisors_bad_lm, incisor_ref)
    # shape_viewer_bad.update_shapes_ref()
    # shape_viewer_bad.update_shapes_all()

    print "Bad incisors and their ssd: "
    for incisor in incisors_bad_lm:
        print incisor.ssd

    training_set = create_training_set(incisors_good_lm)
    eval, evec, mu = principal_component_analysis(training_set, 10)

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
