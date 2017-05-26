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
        num_pc = n - 1

    # Compute average
    mu = np.mean(X, axis=1)

    # Compute correlation matrix C
    C = np.zeros((d, d))
    for i in range(n):
        x_diff = (X[:, i] - mu).reshape(d, 1)
        C = C + np.dot(x_diff, x_diff.T)
    C = C / (n - 1)

    # Compute eigenvalues and eigenvectors
    # eigen_values, eigen_vectors = scipy.sparse.linalg.eigs(C, k=num_pc, which='LM')
    eigen_values, eigen_vectors = eigs(C, k=num_pc, which='LM')
    eigen_values = np.real(eigen_values)
    eigen_vectors = np.real(eigen_vectors)

    return eigen_values, eigen_vectors, mu


def create_training_set(shapes_list, shape_ref):
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

    X = X * shape_ref.scale
    return X


def reconstruct_shape_object(shape_mu, P, b):
    """
        x = mu + P*b 
    """
    lm_x = shape_mu.lm_org + np.dot(P, b).reshape(shape_mu.lm_org.shape)
    return ObjectShape(lm_x)


def show_modes(shape_mu, P, eigenvalues):
    plt.figure()
    myLib.move_figure(position="right")
    coef = np.array([-3, 0, 3])
    for i in range(3):
        for j in range(3):
            b = np.zeros_like(eigenvalues)
            b[i] = coef[j] * np.sqrt(eigenvalues[i])
            shape = reconstruct_shape_object(shape_mu, P, b)
            x = shape.lm_org[0, :]
            y = shape.lm_org[1, :]
            axis_num = (i*3) + (j + 1)
            plt.subplot(3, 3, axis_num)
            plt.plot(x, y, 'k.', markersize=6)
            plt.plot(x, y, 'r--')
            plt.title("b" + str(i) + " = " + str(coef[j]) + "*sqrt(e" + str(i) + ")")
            plt.grid()
            plt.axis('equal')
            plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")

    plt.close('all')

    myLib.tic()
    incisors = load_incisors()
    myLib.toc()

    # incisors = incisors[0:5]

    # landmarksOrg = (np.array([[4, 0, 0], [0, 0, 1]])).astype(float) * 1
    landmarksOrg = (np.array([[0, 5, 5, 0], [0, 0, 1, 1]])).astype(float)
    # landmarksOrg = (np.array([[-5, 5, 10, 5, -5, -10], [-5, -5, 0, 5, 5, 0]], dtype=float))  # Hexagon
    # landmarksOrg = (np.array([[-5, -3, 3, 5, 7, 8, 10, 8, 7, 5, 3, -3, -5, -7, -8, -10, -8, -7],
    #                           [-5, -5, -5, -5, -3, -2, 0, 2, 3, 5, 5, 5, 5, 3, 2, 0, -2, -3]], dtype=float))
    #
    landmarks_ref = np.copy(landmarksOrg)
    incisors = create_shapes(5, landmarks_ref)

    incisor_ref = procrustes_analysis(incisors)

    incisors_bad_lm = []
    incisors_good_lm = []
    for incisor in incisors:
        if incisor.ssd > 0.04 * 10:
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

    training_set = create_training_set(incisors, incisor_ref)
    eigenval, eigenvec, lm_mu = principal_component_analysis(training_set, 3)

    incisor_mu = ObjectShape(lm_mu.reshape(2, np.size(lm_mu, 0) / 2))
    a = incisor_mu.lm_org - incisor_ref.lm_org
    print a

    # view_mu = ShapesViewer([incisor_mu], incisor_ref)
    #
    # view_mu.update_shapes_ref()
    # view_mu.update_shapes_all()

    # b = np.zeros_like(eigenval).reshape(len(eigenval), 1)
    # b1 = np.copy(b)
    # b2 = np.copy(b)
    # b3 = np.copy(b)
    #
    # b1[0, 0] = 0 * np.sqrt(eigenval[0])
    # b2[1, 0] = 3 * np.sqrt(eigenval[1])
    # b3[2, 0] = 3 * np.sqrt(eigenval[2])
    #
    # x1 = (lm_mu + np.dot(eigenvec, b1).T).flatten()
    # x2 = (lm_mu + np.dot(eigenvec, b2).T).flatten()
    # x3 = (lm_mu + np.dot(eigenvec, b3).T).flatten()
    #
    # plt.figure()
    # plt.subplot(311)
    # x = x1[0:np.size(x1, 0) / 2]
    # y = x1[np.size(x1, 0) / 2:]
    # plt.plot(x, y)
    # plt.grid()
    # plt.show()
    #
    # plt.subplot(312)
    # x = x2[0:np.size(x1, 0) / 2]
    # y = x2[np.size(x1, 0) / 2:]
    # plt.plot(x, y)
    # plt.grid()
    # plt.show()
    #
    # plt.subplot(313)
    # x = x3[0:np.size(x1, 0) / 2]
    # y = x3[np.size(x1, 0) / 2:]
    # plt.plot(x, y)
    # plt.grid()
    # plt.show()

    show_modes(incisor_mu, eigenvec, eigenval)

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
