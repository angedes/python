import numpy as np
import plotData as pld
import scipy.optimize as scOptim
import scipy
from common import to_homogeneous, project, crossMatrix, crossMatrixInv, huberLoss
import sfm
from math import sin, cos, acos, atan2

def getFromOp(Op, nPoints, nCameras):
    end_points = 3 * nPoints
    x3D = to_homogeneous(np.reshape(np.array(Op[:end_points]), (3, nPoints), order='F'))
    cameras = []
    for i in range(nCameras-1):
        if i == 0:
            theta = Op[end_points]
            phi = Op[end_points + 1]
            t2 = np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
            R2 = scipy.linalg.expm(crossMatrix(Op[3 * nPoints + 2:3 * nPoints + 5]))
            cameras.append(pld.ensamble_T(R2, t2))
        else:
            sci = end_points + 6*i - 1
            t = np.array([Op[sci], Op[sci + 1], Op[sci + 2]])
            R = scipy.linalg.expm(crossMatrix(Op[sci+3:sci+6]))
            cameras.append(pld.ensamble_T(R, t))

    return x3D, cameras

def resBundleProjection(Op, xData, K_c, nPoints, nCameras):
    """
    -input:
    Op: Optimization parameters: this must include a
    parametrization for T_21 (reference 1 seen from reference 2)
    in a proper way and for X1 (3D points in ref 1)
    xData: list(3xnPoints, nCameras) 2D points on image
    K_c: list(3x3, nCameras) Intrinsic calibration matrix
    nPoints: Number of points
    nCameras: Number of cameras
    -output:
    res: residuals from the error between the 2D matched points
    and the projected points from the 3D points
    (2 equations/residuals per 2D point)
    """

    x3D, T_ci_c1_s = getFromOp(Op, nPoints, nCameras)
    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    T_ci_c1_s.insert(0, pld.ensamble_T(np.eye(3,3), np.zeros(3)))
    res = []
    for i in range(nCameras):
        T_ci_c1 = T_ci_c1_s[i]
        P_ci_c1 = K_c[i] @ C @  T_ci_c1
        x_i = project(P_ci_c1, x3D)
        res_ci = (xData[i] - x_i)[0:2, :]
        res.append(res_ci)
    res_np = np.concatenate(res, axis=1).flatten(order='F')
    return res_np


def reconstruction2D(K, C, Ts, p3D):
    x = []
    Ts.insert(0, pld.ensamble_T(np.eye(3, 3), np.zeros(3)))
    for i in range(len(Ts)):
        P_c1_w = K[i] @ C @ Ts[i]
        x_i = project(P_c1_w, p3D)
        x.append(x_i)
    return x


def optimisationByBA(Ks, Ts_ci_c1, xData, p3D_c1, nPoints, nCameras):
    poses = []

    scale_factor = 1 / np.linalg.norm(Ts_ci_c1[0][0:3, 3])
    for i in range(nCameras-1):
        T_ci_c1 = Ts_ci_c1[i]
        T_ci_c1 = sfm.scaleT(T_ci_c1, scale_factor)

        theta_ext = crossMatrixInv(scipy.linalg.logm(T_ci_c1[0:3, 0:3].astype('float64')))
        theta_ext = [i.real if isinstance(i, complex) else i for i in theta_ext]
        if i == 0:
            theta = acos(T_ci_c1[2, 3])
            phi = atan2(T_ci_c1[1, 3], T_ci_c1[0, 3])
            poses += [theta, phi] + theta_ext
        else:
            poses += [T_ci_c1[0, 3], T_ci_c1[1, 3], T_ci_c1[2, 3]] + theta_ext
    poses = np.array(poses)
    p3D = sfm.scalePoints(p3D_c1, scale_factor)
    Op = np.concatenate((p3D[0:3,:].flatten(order='F'), poses))
    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(xData, Ks, nPoints, nCameras), method='trf', jac='3-point', loss='huber', verbose=1)
    x3D_Op, Ts_c1_ci_Op = getFromOp(np.array(OpOptim.x), nPoints, nCameras)
    return Ts_c1_ci_Op, x3D_Op
