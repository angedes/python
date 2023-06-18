import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotData as pld

from triangulation import triangulate
from common import to_homogeneous, euclidean_error
import sfm
import homography as hm


def drawEpipolarLinesFromPoints(im2, F21, points1, limit = 8, title=None):
    """Draws epipolar lines from points points1 and the fundamental matrix F21"""
    plt.figure()

    # Draw Lines
    if title is not None:
        plt.title(title)
    plt.imshow(im2)
    for i in range(min(points1.shape[1], limit)):
        l2 = F21 @ points1[:, i]
        pld.plot2Dline(l2)

    # Draw Epipolar Center
    c = sfm.getEpipolarPoint(F21).reshape(-1, 1)
    pld.plotMarkersImagePoints(c, color='r', marker='+', label='Epipole')
    plt.legend()

    print('Close the figure to continue')
    plt.show(block=False)
    plt.pause(0.001)


def reconstructCameras(E_21, KC, x1, x2):
    """ Returns the second camera pose and the triangulated points in the local frame of the first camera using a
    votation algorithm over the 4 possible states"""
    U, s, Vh = np.linalg.svd(E_21)
    t = U[:, 2]
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    R_p90 = U @ W @ Vh
    if np.linalg.det(R_p90) < 0:
        R_p90 *= -1
    R_m90 = U @ W.T @ Vh
    if np.linalg.det(R_m90) < 0:
        R_m90 *= -1

    motionSolutions = [ensamble_T(R_p90, t),
                       ensamble_T(R_p90, -t),
                       ensamble_T(R_m90, t),
                       ensamble_T(R_m90, -t)]

    T1 = ensamble_T(np.eye(3, 3), np.zeros(3))
    P1 = KC @ T1
    best_solution = -1
    max_votes = -1
    best_p3Ds = None
    n = x1.shape[1]
    for solIndex in range(len(motionSolutions)):
        T2 = motionSolutions[solIndex]
        P2 = KC @ T2
        p3Ds = np.empty((4, n))
        for i in range(n):
            p3Ds[:, i] = triangulate([P1, P2], [x1[:, i], x2[:, i]])

        if __debug__:
            plt.figure()
            plt.title(f'Motion solution {solIndex}')
            ax = plt.axes(projection='3d', adjustable='box')
            pld.drawRefSystem(ax, T1, '-', 'C1')
            pld.drawRefSystem(ax, np.linalg.inv(T2), '-', 'C2')

            ax.set_xlabel('X')
            ax.set_xlabel('Y')
            ax.set_xlabel('Z')
            # Matplotlib does not correctly manage the axis('equal')
            xFakeBoundingBox = np.linspace(0, 3, 2)
            yFakeBoundingBox = np.linspace(0, 3, 2)
            zFakeBoundingBox = np.linspace(0, 3, 2)
            plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

            pld.plotMarkersImagePoints3D(ax, p3Ds, c="red", marker='+', label="Triangulated points")
            plt.legend()

        # Project and count how many of them are in front of the camera
        in_front_of_P1 = ((P1 @ p3Ds)[2, :] > 0)
        in_front_of_P2 = ((P2 @ p3Ds)[2, :] > 0)
        if __debug__:
            print(in_front_of_P1)
            print(in_front_of_P2)
        in_front_of_both = np.logical_and(in_front_of_P1, in_front_of_P2)
        current_votes = in_front_of_both.sum()
        if current_votes > max_votes:
            best_solution = solIndex
            max_votes = current_votes
            best_p3Ds = p3Ds
    if __debug__:
        print(f"Chosen: {best_solution}")
        print("Close to continue...")
    return motionSolutions[best_solution], best_p3Ds


def getScale(T_GT, T):
    """Scales the translation of the T matrices by scale"""
    return np.linalg.norm(T_GT[0:3, 3]) / np.linalg.norm(T[0:3, 3])

def scaleT(T, scale):
    T2 = T.copy()
    T2[0:3, 3] *= scale
    return T2