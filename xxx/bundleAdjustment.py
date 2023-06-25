import numpy as np
import plotData as pld
import scipy.optimize as scOptim
import scipy
from common import to_homogeneous, project, crossMatrix, crossMatrixInv, huberLoss, plotResidual, euclidean_error
from math import sin, cos, acos, atan2
import matplotlib.pyplot as plt
import sfm


def getFromOp(Op, nPoints):
    x3D = to_homogeneous(np.reshape(np.array(Op[:3 * nPoints]), (3, -1), order='F'))
    theta = Op[3 * nPoints]
    phi = Op[3 * nPoints + 1]
    t2 = np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
    R2 = scipy.linalg.expm(crossMatrix(Op[3 * nPoints + 2:3 * nPoints + 5]))
    T_c2_c1 = pld.ensamble_T(R2, t2)
    return x3D, T_c2_c1


def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    -input:
    Op: Optimization parameters: this must include a
    paramtrization for T_21 (reference 1 seen from reference 2)
    in a proper way and for X1 (3D points in ref 1)
    x1Data: (3xnPoints) 2D points on image 1 (homogeneous
    coordinates)
    x2Data: (3xnPoints) 2D points on image 2 (homogeneous
    coordinates)
    K_c: (3x3) Intrinsic calibration matrix
    nPoints: Number of points
    -output:
    res: residuals from the error between the 2D matched points
    and the projected points from the 3D points
    (2 equations/residuals per 2D point)
    """

    x3D, T_c2_c1 = getFromOp(Op, nPoints)
    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    P_c1_w = K_c @ C @ pld.ensamble_T(np.eye(3,3), np.zeros(3))
    P_c2_c1 = K_c @ C @ T_c2_c1
    x1 = project(P_c1_w, x3D)
    x2 = project(P_c2_c1, x3D)
    res_c1 = (x1Data - x1)[0:2, :]
    res_c2 = (x2Data - x2)[0:2, :]

    res = np.concatenate((res_c1, res_c2), axis=1).flatten(order='F')

    return res


def reconstruction2D(K, C, T_c2_c1, p3D_c1):
    P_c1_w = K @ C @ pld.ensamble_T(np.eye(3,3), np.zeros(3))
    P_c2_c1 = K @ C @ T_c2_c1
    x1 = project(P_c1_w, p3D_c1)
    x2 = project(P_c2_c1, p3D_c1)
    return x1, x2

def optimisationByBA(K, T_c2_c1, x1, x2, p3D_c1):
    scale = sfm.getScaleSingle(T_c2_c1)
    T = sfm.scaleT(T_c2_c1, 1/scale)
    p3D = p3D_c1.copy()
    p3D[0:3, :] /= scale
    theta = acos(T[2, 3])
    phi = atan2(T[1, 3], T[0, 3])
    theta_ext = crossMatrixInv(scipy.linalg.logm(T[0:3, 0:3].astype('float64')))
    theta_ext = [i.real if isinstance(i, complex) else i for i in theta_ext]
    Op = np.concatenate((p3D[0:3,:].flatten(order='F'), [theta, phi], theta_ext))
    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(x1, x2, K, x1.shape[1]), method='trf', jac='3-point', loss='huber', verbose=1)
    x3D_Op, T_c2_c1_Op = getFromOp(np.array(OpOptim.x), x1.shape[1])
    return T_c2_c1_Op, x3D_Op


def reconstruction2Dfull(image_pers_2, K, C, T_c2_c1, p3D_c1, x2Data,
                            title="ME error image"):  # Plot the 2D reconstruction
    x1, x2 = reconstruction2D(K, C, T_c2_c1, p3D_c1)
    plt.figure()
    plt.imshow(image_pers_2)
    plotResidual(x2Data, x2, '-')
    print(f"{title}: {euclidean_error(x2Data, x2)}")