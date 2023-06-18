import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

import cv2
import plotData as pld
from common import project


def buildDLTEquation(p2D, p3D):

    equation = [[-p3D[0], -p3D[1], -p3D[2], -p3D[3], 0, 0, 0, 0, p2D[0] * p3D[0], p2D[0] * p3D[1], p2D[0] * p3D[2],
                 p2D[0] * p3D[3]],
                [0, 0, 0, 0, -p3D[0], -p3D[1], -p3D[2], -p3D[3], p2D[1] * p3D[0], p2D[1] * p3D[1], p2D[1] * p3D[2],
                 p2D[1] * p3D[3]]]

    return equation

def decomposeP(est_P):

    canonical = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    est_K, est_R = np.linalg.qr(est_P[:, :3])
    D = np.diag(np.sign(np.diag(est_K)))
    if np.linalg.det(D) < 0:
        D[1,1] *= -1
    K = est_K @ D

    u, s, v = np.linalg.svd(est_P)
    est_C = v[-1] / v[-1, -1]
    R = D @ est_R
    est_C_T = est_C[0:3].reshape(3, 1)
    est_t = -R @ est_C_T
    newrow = [0, 0, 0, 1]
    T = np.concatenate((R, est_t), axis=1)
    T = np.vstack([T, newrow])
    KC = K @ canonical
    P = KC @ T

    """
    u, s, v = np.linalg.svd(est_P)
    est_C = v[-1] / v[-1, -1]
    est_K, est_R = linalg.rq(est_P[:, :3])
    D = np.diag(np.sign(np.diag(est_K)))
    if np.linalg.det(D) < 0:
        D[1, 1] *= -1
    K_line = D @ est_K
    K = K_line / K_line[2, 2]
    """

    return P, K, T


def dlt(p2Ds, p3Ds):

    rows1, cols1 = np.shape(p2Ds)
    rows2, cols2 = np.shape(p3Ds)

    assert (rows1 == rows2)
    linear_system = np.empty([rows1*2, 12])

    # Add 2 rows per point
    for i in range(rows1):
        # 2 rows from same point
        linear_system[[2 * i, 2 * i + 1], :] = buildDLTEquation(p2Ds[i, :], p3Ds[i, :])
    u, s, vh = np.linalg.svd(linear_system)
    est_P = vh[-1]

    est_P = est_P.reshape(3, 4)

    P, K, T = decomposeP(est_P)

    return P, K, T


def main():
    im3 = cv2.cvtColor(cv2.imread("old.png"), cv2.COLOR_BGR2RGB)
    old = cv2.resize(im3, dsize=[640, 480])

    points3D = np.loadtxt('points_3D_old.txt')
    points2D = np.loadtxt('points_2D_old.txt')

    x3D = points3D.T
    x2D = points2D.T


    # DLT

    P, K, T = dlt(x2D, x3D)

    proj_points = project(P, points3D)

    proj_points[:,-1] = proj_points[:, -2]

    rows1, cols1 = np.shape(proj_points)

    index1 = []

    for j in range(cols1 - 1):
        if proj_points[0, j] < 0 or proj_points[0, j] > 640 or proj_points[1, j] < 0 or \
                proj_points[1, j] > 480:
            index1.append(j)

    proj_points = np.delete(proj_points, index1, axis=1)

    fig = plt.figure()
    plt.imshow(old)
    pld.plotMarkersImagePoints(proj_points, color='b', marker='+')
    plt.show()


if __name__ == '__main__':
    main()