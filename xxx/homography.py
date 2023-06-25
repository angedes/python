import plotData as pld
import numpy as np
import matplotlib.pyplot as plt
from common import computeCholeskyFromPoints, computeEqSys, euclidean_error

def normalizePlane(P):
    """Normalizes the input plane T"""
    normfactor = np.sqrt((P[0]**2 + P[1]**2 + P[2]**2))
    n = P[0:3]
    n = np.expand_dims(n, axis=0).T
    return n / normfactor, P[3] / normfactor


def homographyFromPoses(T21, K1, K2, PI1):
    """Calculates the homography from T poses and calibration matrices"""
    t21 = T21[0:3, 3]
    t21 = np.expand_dims(t21, axis=0).T
    R21 = T21[0:3, 0:3]

    n, d = normalizePlane(PI1)

    return K2 @ (R21 - t21 @ n.T / d) @ np.linalg.inv(K1)


def visualizeHomography(H21, im2, x1, x2, title=None):
    """Visualizes the homography by the set of points siven in im2"""
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(im2)
    x2_h = H21 @ x1
    x2_h /= x2_h[2]
    pld.plotMarkersImagePoints(x2, color='r', marker='+', label='Ground truth')
    pld.plotMarkersImagePoints(x2_h, color='b', marker='+', label='Homography')
    plt.legend()

    return x2_h

def homography_error(x2, x2_h, plot=False):
    """Calculates the error from a homography given the ground truth x2 and the homographies ones x2_h.
    Plots the typical direction error (*50 to be able to see something) if requested."""
    euclidean_error_h = euclidean_error(x2, x2_h)
    difVect = x2_h - x2
    difVect = (difVect / np.linalg.norm(difVect))
    averageVect = np.mean(difVect, axis=1)
    if plot:
        for j in range(x2.shape[1]):
            originPt = x2[:, j]
            destPt = originPt + averageVect * euclidean_error_h * 50
            plt.plot([originPt[0], destPt[0]], [originPt[1], destPt[1]], '--', 'k', 1, color='green')
    return euclidean_error_h, averageVect


def prepareHeqsys(x0, x1):
    """Prepares the equation system for computing the homography matrix from matches x0-x1"""
    n = x0.shape[1]
    A = np.empty((2*n, 9))
    # Every match
    for j in range(n):
        x0p = x0[:, j]
        x1p = x1[:, j]
        A[2*j, :] = np.array([x1p[2]*x0p[0], x1p[2]*x0p[1], x1p[2]*x0p[2], 0, 0, 0, -x1p[0]*x0p[0], -x1p[0]*x0p[1], -x1p[0]*x0p[2]])
        A[2*j+1, :] = np.array([0, 0, 0, x1p[2]*x0p[0], x1p[2]*x0p[1], x1p[2]*x0p[2], -x1p[1]*x0p[0], -x1p[1]*x0p[1], -x1p[1]*x0p[2]])
    return A

def computeH(x0, x1, normalize=False):
    """Computes the homography matrix from matches x0-x1 and normalizes using Cholesky if necessary"""
    if normalize:
        T0 = computeCholeskyFromPoints(x0)
        T1 = computeCholeskyFromPoints(x1)
        x0 = T0 @ x0
        x1 = T1 @ x1
    A = prepareHeqsys(x0, x1)
    H = np.reshape(computeEqSys(A), (3,3))
    # Denormalize
    if normalize:
        H = T1.T @ H @ T0
    H /= H[2, 2]
    return H

