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
from ProcrustesAnalysis import separate_good_bad_shape_fit

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
    coef = np.array([-2, 0, 2])
    for i in range(3):
        for j in range(3):
            b = np.zeros_like(eigenvalues)
            b[i] = coef[j] * np.sqrt(eigenvalues[i])
            shape = reconstruct_shape_object(shape_mu, P, b)
            x = shape.lm_org[0, :]
            y = shape.lm_org[1, :]
            axis_num = (i * 3) + (j + 1)
            plt.subplot(3, 3, axis_num)
            plt.plot(x, y, 'k.', markersize=6)
            plt.plot(x, y, 'r--')
            plt.title("b" + str(i) + " = " + str(coef[j]) + "*sqrt(e" + str(i) + ")")
            plt.grid()
            plt.axis('equal')
            plt.show()


def save_PCA_results(directory_path, shape_name, shape_idx, eigenvectors, eigenvalues, mean):
    myLib.ensure_dir(directory_path)
    file_path_eigenvectors = directory_path + shape_name + "_" + str(shape_idx) + "_eigenvectors"
    file_path_eigenvalues = directory_path + shape_name + "_" + str(shape_idx) + "_eigenvalues"
    file_path_mean = directory_path + shape_name + "_" + str(shape_idx) + "_mean"

    np.save(file_path_eigenvectors, eigenvectors)
    np.save(file_path_eigenvalues, eigenvalues)
    np.save(file_path_mean, mean)

if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")

    plt.close('all')

    incisor_idx_list = np.arange(1,9)

    for incisor_idx in incisor_idx_list:
        myLib.tic()
        # incisor_idx = 8
        incisors = load_incisors([incisor_idx])
        # incisors = load_incisors([5, 6, 7, 8])
        myLib.toc()

        # incisors = incisors[0:5]

        # incisors = create_shapes(6)

        incisor_ref = procrustes_analysis(incisors)

        incisors_good_fit, incisors_bad_fit = separate_good_bad_shape_fit(incisors)

        # shape_viewer_good = ShapesViewer(incisors_good_fit, incisor_ref, "good shapes")
        # shape_viewer_good.update_shapes_ref()
        # shape_viewer_good.update_shapes_all()
        #
        # shape_viewer_bad = ShapesViewer(incisors_bad_fit, incisor_ref, "bad shapes")
        # shape_viewer_bad.update_shapes_ref()
        # shape_viewer_bad.update_shapes_all()

        training_set = create_training_set(incisors_good_fit, incisor_ref)
        eigenval, eigenvec, lm_mu = principal_component_analysis(training_set, 3)
        print "\neigenvalues = " + str(eigenval)

        incisor_mu = ObjectShape(lm_mu.reshape(2, np.size(lm_mu, 0) / 2))
        # diff_mean = incisor_mu.lm_org - incisor_ref.lm_org
        # # print "\ndiff_mean:"
        # # print str(diff_mean)
        #
        # shape_viewer_mu = ShapesViewer([incisor_mu], incisor_ref)
        # shape_viewer_mu.update_shapes_ref()
        # shape_viewer_mu.update_shapes_all()

        # show_modes(incisor_mu, eigenvec, eigenval)

        # Save PCA
        dir_path = "Project_Data/_Data/PCA/"
        # save_PCA_results(dir_path, "incisor", incisor_idx, eigenvec, eigenval, lm_mu)

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
