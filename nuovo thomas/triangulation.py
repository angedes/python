import numpy as np

def buildTriangulateEquation(P, p2D):
    """Builds a triangulation equation given a projection matrix and its 2D point"""
    equation = np.empty((2, 4))
    for k in range(2):
        for i in range(4):
            equation[k, i] = P[2, i] * p2D[k] - P[k, i]
    return equation

def triangulate(Ps, p2Ds):
    """Triangulates a point into 3D given the projection matrices Ps and its corresponding 2D point in p2Ds"""
    assert(len(Ps) == len(p2Ds))
    linear_system = np.empty((len(Ps) * 2, 4))
    # Add 2 rows per point
    for i in range(len(p2Ds)):
        # 2 rows from same point
        linear_system[[2*i, 2*i+1], :] = buildTriangulateEquation(Ps[i], p2Ds[i])
    u, s, vh = np.linalg.svd(linear_system)
    p3D = vh[-1, :]
    p3D /= p3D[3]
    return p3D
