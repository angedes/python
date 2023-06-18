import numpy as np
from common import *
from triangulation import *
import cv2

# to remove
if __debug__:
    import matplotlib.pyplot as plt
    import plotData as pld


def compute3D_sfm(x1_matches, x2_matches, K, KC):
    F = computeF(x1_matches, x2_matches)

    E = K.T @ F @ K

    T, p3Ds = reconstructCameras(E, KC, x1_matches, x2_matches)
    scale = 1
    p3Ds[0:3] *= scale
    p3Ds = np.delete(p3Ds, 99, axis=1)
    T = scaleT(T, scale)
    T = np.linalg.inv(T)

    ax = pld.plotWorldFromC1(T, p3Ds, title="3D points from C1 with SfM")
    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)

    plt.show()

    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    return T, p3Ds


def fundamentalFromPoses(Twc1, Twc2, K1, K2):
    """Computes the fundamental matrix from the camera poses and the calibration matrices"""
    T21 = np.linalg.inv(Twc2) @ Twc1
    t = T21[0:3, 3]
    R = T21[0:3, 0:3]
    E = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ]) @ R
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def prepareFeqsys(x0, x1):
    """Prepares the equation system for computing the fundamental matrix from matches x0-x1"""
    n = x0.shape[1]
    A = np.empty((n, 9))
    # Every match
    for j in range(n):
        # Iterate over axis
        for i1 in range(3):
            for i0 in range(3):
                A[j, i1*3+i0] = x0[i0, j] * x1[i1, j]
    return A


def computeF(x0, x1, normalize=False):
    """Computes the fundamental matrix from matches x0-x1 and normalizes using Cholesky if necessary"""
    if normalize:
        T0 = computeCholeskyFromPoints(x0)
        T1 = computeCholeskyFromPoints(x1)
        x0 = T0 @ x0
        x1 = T1 @ x1
    A = prepareFeqsys(x0, x1)
    F_flat = computeEqSys(A)
    # Reshape row matrix to 3x3
    F_hat = np.reshape(F_flat, (3, 3))
    # Enforce rank 2 to fit the data
    F = enforceRank(F_hat, 2)
    # Denormalize
    if normalize:
        F = T1.T @ F @ T0

    return F

def scalePoints(P, scale):
    Ps = P.copy()
    Ps[0:3, :] *= scale
    return Ps

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
        """
        if __debug__:
            print(in_front_of_P1)
            print(in_front_of_P2)
        """
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

def scaleT(T, scale):
    """Scales the translation of the T matrices by scale"""
    T2 = np.copy(T)
    T2[0:3, 3] *= scale
    return T2

def getEpipolarPoint(F_21):
    """
    Returns the Epipolar point e_1 (Where Camera 1 is in Camera 2's view)

    Transpose the Fundamental Matrix to obtain the opposite point (Where Camera 2 is in Camera 1's view)
    """
    #print("F^T:\n", F_21.T)
    c = computeEqSys(F_21.T)
    #print("c:\n", c)
    c /= c[2]
    #print("c (Normalized):\n", c)
    return c

def getScaleSingle(T):
    return np.linalg.norm(T[0:3, 3])

def recoverPose(x3, x, K):
    imagePoints = np.ascontiguousarray(x3[0:2, :].T).reshape((x3.shape[1], 1, 2))
    objectPoints = np.ascontiguousarray(x[0:3, :].T).reshape((x3.shape[1], 1, 3))
    retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, K, None, flags=cv2.SOLVEPNP_EPNP)
    rvec = rvec.flatten()
    tvec = tvec.flatten()
    T_c3_c1 = ensamble_T(scipy.linalg.expm(crossMatrix(rvec)), tvec)
    return T_c3_c1