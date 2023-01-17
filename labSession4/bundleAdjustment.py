from plotData import *
from scipy.linalg import expm, logm
import math
import scipy.optimize as scOptim
from plotGroundTruth import plotResidual

def crossMatrixInv(M):
    x = [M[2,1],M[0,2],M[1,0]]
    return x

def crossMatrix(x):
    M = np.array([[0,-x[2],x[1]],
                 [x[2],0,-x[0]],
                 [-x[1],x[0],0]])

    return M

def translationToAngle(t):

    theta = math.asin(-t[1])
    phi = math.atan2(t[2],t[0])

    return np.array([theta,phi])

def angleToTranslation(th,phi):

    y = -math.sin(th)
    z = math.cos(th)*math.sin(phi)
    x = math.cos(th)*math.cos(phi)

    return np.array([x,y,z])

def codeOp(T,X,nCameras = 2,t_ops=2):

    """
    Code the parameters for bundle adjustment
    T:  Transformation matrix/matrices
        T must be shape (nCameras-1,4,4)
    X: Homogeneous 3D points, 1 dimension per row.
    nCameras: Number of cameras.
    t_ops: Number of descriptors for the second camera translation. Either 2 or 3.
    """

    R = T[0, 0:3, 0:3]
    th = np.array(crossMatrixInv(logm(R.astype("float64"))))

    t = T[0, 0:3, 3]
    inc, az = translationToAngle(t)
    if t_ops == 2:
        op = [th[0],th[1],th[2],inc,az]
    elif t_ops == 3:
        op = [th[0],th[1],th[2],t[0],t[1],t[2]]

    for i in range(1,nCameras-1):
        R = T[i,0:3,0:3]
        th = np.array(crossMatrixInv(logm(R.astype("float64"))))

        t = T[i,0:3,3]
        dist = np.linalg.norm(t)
        t/=dist
        inc, az = translationToAngle(t)
        op = np.concatenate((op,[th[0],th[1],th[2],inc,az,dist]))

    points = np.array(X[:3]).flatten()

    op = np.concatenate([op,points])

    return op

def decodeOp(op,nCameras=2,t_ops=2):

    R2 = expm(crossMatrix(np.array(op[0:3])))
    if t_ops==2:
        t2 = angleToTranslation(op[3],op[4])
        i = 5
    elif t_ops ==3:
        t2 = np.array([op[3:6]]).flatten()
        i = 6

    T2 = ensamble_T(R2, t2)
    Ts = np.array([T2])

    for j in range(1,nCameras-1):
        R = expm(crossMatrix(np.array(op[i:i+3])))
        t = angleToTranslation(op[i+3],op[i+4])*op[i+5]
        T = ensamble_T(R,t)
        Ts = np.concatenate([Ts,[T]])
        i += 6

    points = np.array(op[i:])
    X = np.ones((4,len(points)//3))
    X[:3] = points.reshape(3,len(points)//3)

    return Ts,X

def makePfromT(T,K):

    I_3_4 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])

    return K@I_3_4@T

def project(X,P):

    x = P@X
    x /= x[2]

    return x

def resBundleProjection(op,x1,x2,K_c,nPoints,t_ops=2):
    """
    Compute residuals (deltax,deltay) from the projection matrix, obtained from op.
    Op represents T_c2_c1=[R,t] as 5 parameters or 6, as well as the triangulated points.
    """

    T_c2_c1, X = decodeOp(op,t_ops)
    P1 = makePfromT(np.identity(4),K_c)
    P2 = makePfromT(T_c2_c1.squeeze(),K_c)

    # Project the points
    x1_p = project(X,P1)[:2]
    x2_p = project(X,P2)[:2]

    # Calculate residuals
    x1_res = x1_p - x1
    x2_res = x2_p - x2

    delta1_x = x1_res[0]
    delta1_y = x1_res[1]
    delta2_x = x2_res[0]
    delta2_y = x2_res[1]

    res = np.concatenate([delta1_x,delta1_y,delta2_x,delta2_y]).flatten()

    return res

def resBundleProjectionMultiCam(op,X_img,K_c,nCameras,nPoints,t_ops=2):

    Ts, X = decodeOp(op,nCameras,t_ops)

    I_3_4 = [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0]]

    T_c2_c1 = Ts[0]

    Ps = np.zeros((nCameras,3,4))

    Ps[0] = K_c@I_3_4@np.identity(4) # P1
    Ps[1] = K_c@I_3_4@T_c2_c1 # P2

    for i in range(2,nCameras):

        j = t_ops+3+(i-2)*6 # first descriptor of camera i
        R = expm(crossMatrix(op[j:j+3]))
        t = angleToTranslation(op[j+3],op[j+4])*op[j+5]
        T = ensamble_T(R,t)
        Ps[i] = K_c@I_3_4@T

    # Project the points and calculate the residuals

    res = []

    for P, i in zip(Ps,range(nCameras)):

        x = project(X,P)

        delta_x = x[0]-X_img[i][0]
        delta_y = x[1]-X_img[i][1]

        res = np.concatenate([res,delta_x,delta_y])

    return res

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # Read the images
    path_image_1 = 'image1.png'
    path_image_2 = 'image2.png'
    path_image_3 = 'image3.png'
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    image_pers_3 = cv2.imread(path_image_3)

    # Calibration
    K_c = np.loadtxt('K_c.txt')
    X_w = np.loadtxt('X_w.txt')

    # Load points on each image
    # Shape: 2 rows, 103 columns. Each column is an (x,y) pair for a point.

    x1 = np.loadtxt('x1Data.txt')
    x2 = np.loadtxt('x2Data.txt')
    x3 = np.loadtxt('x3Data.txt')

    # ---------------------------------------
    # QUESTION 2
    # ---------------------------------------

    # FOR NOW, COPY-PASTE THE RESULT TRANSFORMATION MATRIX AND 3D POINTS FROM LAB2

    # Transformation matrices, 4x4
    T_c1_c2 = np.load("T_21.npz")
    T_c1_c2 = T_c1_c2.f.arr_0
    T_c2_c1 = np.linalg.inv(T_c1_c2)

    T_w_c1_gt = np.loadtxt("T_w_c1.txt")
    T_w_c2_gt = np.loadtxt("T_w_c2.txt")
    T_c2_c1_gt = np.linalg.inv(T_w_c2_gt)@T_w_c1_gt
    # Triangulated points from SfM, 4x103
    X_tri_c1 = np.load("X_tri_c1.npz")
    X_tri_c1 = X_tri_c1.f.arr_0

    # Project initial triangulated points

    P2 = makePfromT(T_c2_c1,K_c)
    P1 = makePfromT(np.identity(4),K_c)

    # Projected points

    X_1 = project(X_tri_c1,P1)
    X_2 = project(X_tri_c1, P2)

    # Compute initial guess
    op = codeOp(np.array([T_c2_c1]),X_tri_c1)
    print(op)

    nPoints = X_tri_c1.shape[1]
    res = resBundleProjection(op,x1,x2,K_c,nPoints) #(This does the same as the 4 lines below)
    #X = np.zeros((2,x1.shape[0],x1.shape[1]))
    #X[0] = x1
    #X[1] = x2
    #res = resBundleProjectionMultiCam(op, X, K_c, 2, nPoints)

    print("Initial residual L2:",np.linalg.norm(res))

    # Optimize

    opOptim = scOptim.least_squares(resBundleProjection, op, args=(x1,x2,K_c,nPoints), method='trf')
    print(opOptim.x)

    res = resBundleProjection(opOptim.x,x1,x2,K_c,nPoints)
    print("Optimised residual L2:",np.linalg.norm(res))

    T_c2_c1_op, X_op = decodeOp(opOptim.x)
    T_c2_c1_op = T_c2_c1_op[0]
    T_c1_c2_op = np.linalg.inv(T_c2_c1_op)

    P2_op = makePfromT(T_c2_c1_op,K_c)

    # Draw result in 3d

    fig3D = plt.figure(1)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, T_c1_c2, '-', 'C2')
    drawRefSystem(ax, T_c1_c2_op, '-', 'C2-op')
    ax.scatter(X_op[0, :], X_op[1, :], X_op[2, :], marker='.')
    ax.scatter(X_tri_c1[0, :], X_tri_c1[1, :], X_tri_c1[2, :], marker='x')
    plt.title('Image 1 - Lab2 in yellow, LS in blue')

    plt.show()

    # Re-project

    x1_optim = project(X_op,P1)
    x2_optim = project(X_op,P2_op)

    # Plot the 2D points
    plt.figure(1)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1, x1_optim, 'k-')
    plotResidual(x1, X_1, 'y-')
    plt.plot(x1_optim[0, :], x1_optim[1, :], 'b.')
    plt.plot(X_1[0, :], X_1[1, :], 'y.')
    plt.plot(x1[0, :], x1[1, :], 'rx')
    #plotNumberedImagePoints(x1Data[0:2, :], 'r', 4)
    plt.title('Image 1 - GT in red, Lab2 in yellow, LS in blue')
    plt.draw()

    plt.figure(2)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2, x2_optim, 'k-')
    plt.plot(x2_optim[0, :], x2_optim[1, :], 'b.')
    plt.plot(X_2[0, :], X_2[1, :], 'y.')
    plt.plot(x2[0, :], x2[1, :], 'rx')
    # plotNumberedImagePoints(x1Data[0:2, :], 'r', 4)
    plt.title('Image 2 - GT in red, Lab2 in yellow, LS in blue')
    plt.draw()

    plt.show()

    # ----------------------------------------
    # QUESTION 3
    # ----------------------------------------

    imagePoints = np.ascontiguousarray(x3[0:2,:].T).reshape((x3.shape[1], 1, 2))
    objectPoints = np.ascontiguousarray(X_op[0:3,:].T).reshape((X_op.shape[1],1,3))

    retval, r_c3_c1, t_c3_c1 = cv2.solvePnP(objectPoints, imagePoints, K_c, None, flags=cv2.SOLVEPNP_EPNP)

    R_c3_c1 = expm(crossMatrix(r_c3_c1))
    T_c3_c1 = ensamble_T(R_c3_c1,t_c3_c1.reshape((3,)))
    T_c1_c3 = np.linalg.inv(T_c3_c1)
    P_c3_c1 = makePfromT(T_c3_c1,K_c)
    X_3 = project(X_op,P_c3_c1)

    # Draw result in 3d

    fig3D = plt.figure(4)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, T_c1_c3, '-', 'C3')
    ax.scatter(X_op[0, :], X_op[1, :], X_op[2, :], marker='.')
    plt.title('PnP solution for camera 3')

    plt.show()

    # ----------------------------
    # QUESTION 4
    # ----------------------------

    # Compute initial guess

    nCameras = 3
    op = codeOp(np.array([T_c2_c1_op,T_c3_c1]),X_op,nCameras=nCameras)
    print(op)

    X = np.zeros((nCameras,x1.shape[0],x1.shape[1]))
    X[0] = x1
    X[1] = x2
    X[2] = x3

    res = resBundleProjectionMultiCam(op, X, K_c, nCameras, nPoints)

    print("Initial residual L2:",np.linalg.norm(res))

    opOptim = scOptim.least_squares(resBundleProjectionMultiCam, op, args=(X, K_c, nCameras, nPoints), method='trf')

    print(opOptim.x)

    res = resBundleProjectionMultiCam(opOptim.x, X, K_c, 3, nPoints)

    print("Optimised residual L2:",np.linalg.norm(res))

    Ts, X_op3 = decodeOp(opOptim.x,nCameras=nCameras)

    T_c2_c1_op2 = Ts[0]
    T_c1_c2_op2 = np.linalg.inv(T_c2_c1_op2)
    P2_op = makePfromT(T_c2_c1_op,K_c)
    T_c3_c1_op = Ts[1]
    T_c1_c3_op = np.linalg.inv(T_c3_c1_op)
    P3_op = makePfromT(T_c3_c1_op,K_c)

    # Draw result in 3d

    fig3D = plt.figure(5)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, T_c1_c2_op2, '-', 'C2-op')
    drawRefSystem(ax, T_c1_c2_op2, '-', 'C2-op(3view)')
    drawRefSystem(ax, T_c1_c3_op, '-', 'C3-op')
    drawRefSystem(ax, T_c1_c3, '-', 'C3-PnP')
    ax.scatter(X_op[0, :], X_op[1, :], X_op[2, :], marker='.')
    ax.scatter(X_op3[0, :], X_op3[1, :], X_op3[2, :], marker='x')

    plt.show()

    # Re-project
    x1_optim = project(X_op3,P1)
    x2_optim = project(X_op3,P2_op)
    x3_optim = project(X_op3,P3_op)

    # Plot the 2D points
    plt.figure(6)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1, x1_optim, 'k-')
    plotResidual(x1, X_1, 'y-')
    plt.plot(x1_optim[0, :], x1_optim[1, :], 'b.')
    plt.plot(X_1[0, :], X_1[1, :], 'y.')
    plt.plot(x1[0, :], x1[1, :], 'rx')
    plt.title('Image 1 - GT in red, Lab2 in yellow, LS in blue')
    plt.draw()

    plt.figure(7)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2, x2_optim, 'k-')
    plt.plot(x2_optim[0, :], x2_optim[1, :], 'b.')
    plt.plot(X_2[0, :], X_2[1, :], 'y.')
    plt.plot(x2[0, :], x2[1, :], 'rx')
    plt.title('Image 2 - GT in red, Lab2 in yellow, LS in blue')
    plt.draw()

    plt.figure(8)
    plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
    plotResidual(x3, x3_optim, 'k-')
    plt.plot(x3_optim[0, :], x3_optim[1, :], 'b.')
    plt.plot(X_3[0, :], X_3[1, :], 'y.')
    plt.plot(x3[0, :], x3[1, :], 'rx')
    plt.title('Image 3 - GT in red, Lab2 in yellow, LS in blue')
    plt.draw()

    plt.show()
