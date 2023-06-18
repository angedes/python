import numpy as np
import scipy
from math import sqrt
import matplotlib.pyplot as plt

def project(P, x3d):
    x = P @ x3d
    x /= x[2, :]
    return x

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4), dtype=np.float32)
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def computeCholeskyFromPoints(ps):
    """Computes the Cholesky decomposition from points ps"""
    n = ps.shape[1]
    A = np.zeros((3, 3))
    for i in range(n):
        p = np.expand_dims(ps[:, i], axis=0).T
        A += p @ p.T
    A /= n
    N = np.linalg.cholesky(A)
    return np.linalg.inv(N)


# Enforce rank by setting to 0 the little eigenvalues
def enforceRank(M, rank):
    """Enforces a certain rank to the matrix M by setting to 0 the last eigenvalues."""
    u, s, vh = np.linalg.svd(M)
    s[rank:] = 0.0
    return u @ np.diag(s) @ vh


def to_homogeneous(ps):
    """Convert the points ps to homogeneous coordinates"""
    n = ps.shape[1]
    ones = np.ones((1, n))
    return np.vstack((ps, ones))


def computeEqSys(A):
    """Compute a homogeneous system of equations using SVD and returning the eigenvectors associated"""
    u, s, vh = np.linalg.svd(A)
    sol = vh[-1, :]
    return sol


def euclidean_error(actual, predicted):
    """ Computes the Mean of the translation error as euclidean distance"""
    return np.mean(np.sqrt(np.sum(np.square(predicted - actual), axis=0)))

def euclidean_error_pair(actual, predicted):
    """ Computes the translation error as euclidean distance"""
    return np.sqrt(np.sum(np.square(predicted - actual), axis=0))

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def calculateTransformsDiff(T1, T2):
    # Positional Error (Meters)
    err_pos = np.sqrt(np.sum((T1[0:3, 3] - T2[0:3, 3])**2))

    # Rotational Error (Geodesic Distance)
    R_mult = crossMatrixInv(scipy.linalg.logm((T1[0:3, 0:3].T @ T2[0:3, 0:3]).astype('float64')))
    err_rot = np.linalg.norm(R_mult)

    return err_pos, err_rot

def crossMatrix(x):
    M = np.array([[0,   -x[2], x[1]],
                 [x[2],    0, -x[0]],
                 [-x[1], x[0],   0]], dtype="object")
    return M

def huberLoss(res, a):
    hub_res = res.copy()
    a2 = a * a
    for i in range(res.shape[0]):
        # Huber function
        s = res[i]
        if s > a2:
            hub_res[i] = 2 * a2 * sqrt(s / a2) - 1
    return hub_res

def euclidean_error(actual, predicted):
    """ Computes the Mean of the translation error as euclidean distance"""
    return np.mean(np.sqrt(np.sum(np.square(predicted - actual), axis=0)))

def euclidean_distance(actual, predicted):
    """ Computes the Mean of the translation error as euclidean distance"""
    actualN = np.delete(actual, 2)
    predictedN = np.delete(predicted, 2)
    return np.linalg.norm(actualN - predictedN)

def euclidean_distance3D(actual, predicted):
    """ Computes the Mean of the translation error as euclidean distance"""
    actualN = np.delete(actual, 3)
    predictedN = np.delete(predicted, 3)
    return np.linalg.norm(actualN - predictedN)

def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

