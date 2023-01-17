#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Homography, Fundamental Matrix and Two View SfM
#
# Date: 16 September 2022
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
# Assignment completed by: Angelo Desiato, Miguel Marcos, Jorge Pina
#
# Version: 1.0
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4), dtype=np.float32)
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)

def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

# --------------------
# OUR CODE STARTS HERE
# --------------------

def drawEpipLine(F,x):

    l = F@x
    print("Drawn line:",l)

    p1 = np.array([0,-l[2]/l[1]])
    p2 = np.array([-l[2]/l[0],0])

    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.set_title('Clicked epipolar line:')
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.axline(p1,p2,linewidth=2, color='g')
    plt.draw()
    plt.show()

def interactiveEpip(F):

    img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.set_title('Click a point to draw the epipolar line')
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)

    def onclick(event):
        print('Click: x=%f, y=%f' % (event.xdata, event.ydata))
        p_x = event.xdata
        p_y = event.ydata
        drawEpipLine(F,(p_x,p_y,1))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

def getFundamental(T_w_c1,T_w_c2,K1,K2):

    T_c1_w = np.linalg.inv(T_w_c1)
    T_c1_c2 = T_c1_w@T_w_c2

    R = T_c1_c2[0:3,0:3]
    t = [[0,-T_c1_c2[2,3],T_c1_c2[1,3]],
         [T_c1_c2[2,3],0,-T_c1_c2[0,3]],
         [-T_c1_c2[1,3],T_c1_c2[0,3],0]]

    E = t@R
    F = np.transpose(np.linalg.inv(K1))@E@np.linalg.inv(K2)
    F = F/F[2,2]

    return F,E,T_c1_c2

def approxFundamental(p_1,p_2):

    M = np.zeros((p_1.shape[1],9))

    for i in range(p_1.shape[1]):
        x_0 = p_1[0,i]
        x_1 = p_2[0,i]
        y_0 = p_1[1,i]
        y_1 = p_2[1,i]

        M[i] = [x_0*x_1,y_0*x_1,x_1,x_0*y_1,y_0*y_1,y_1,x_0,y_0,1]

    u, s, vh = np.linalg.svd(M)

    vh = vh[-1]

    F = np.array([vh[0:3],vh[3:6],vh[6:9]])

    u,s,vh = np.linalg.svd(F)
    s[-1] = 0
    s = np.diag(s)
    F = u@s@vh
    F = F/F[2,2]

    return F

def getEfromF(F,K0,K1):

    E = np.transpose(K1)@F@K0

    # Proof of correct computation of E
    # F = np.transpose(np.linalg.inv(K1))@E@np.linalg.inv(K0)
    # print(F)

    return E

def SfM(E,K1,K2,x1,x2,plot=False,img1=None,img2=None):

    u, s, vh = np.linalg.svd(E)

    t = np.array(u[:,-1])
    w = np.array([[0,-1,0],
                [1,0,0],
                [0,0,1]])

    R_plus90 = u@w@vh
    R_minus90 = u@np.transpose(w)@vh

    if np.linalg.det(R_plus90)<0:
        R_plus90 = -R_plus90
    if np.linalg.det(R_minus90)<0:
        R_minus90 = -R_minus90

    if plot:
        fig3D = plt.figure()

        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_title('Structure from motion. C1 as W. C2 fulfills chirality condition')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    I3 = np.identity(3)
    I_3_4 = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0]])

    T_1 = ensamble_T(I3,np.zeros((3,)))
    P_1 = K1@I_3_4@T_1

    T_12s = np.array([ensamble_T(R_plus90,t),ensamble_T(R_plus90,-t),ensamble_T(R_minus90,t),ensamble_T(R_minus90,-t)])

    T_21s = np.copy(T_12s) # (Just for the shape of the array, overwritten below)
    for i in range(4):
        T_21s[i] = np.linalg.inv(T_12s[i])

    P_2s = [K2@I_3_4@T_21s[0],K2@I_3_4@T_21s[1],K2@I_3_4@T_21s[2],K2@I_3_4@T_21s[3]]

    best_P = P_2s[0]
    best_T = T_12s[0]
    best_score = 0
    best_tri = None

    for T_12,T_21,P_2 in zip(T_12s,T_21s,P_2s):

        score = 0
        X_tri = np.ones((4,x1.shape[1]))

        for p in range(x1.shape[1]):
            xi = x1[0, p]
            yi = x1[1, p]
            eq1 = [xi * P_1[2, 0] - P_1[0, 0], xi * P_1[2, 1] - P_1[0, 1], xi * P_1[2, 2] - P_1[0, 2],
                   xi * P_1[2, 3] - P_1[0, 3]]
            eq2 = [yi * P_1[2, 0] - P_1[1, 0], yi * P_1[2, 1] - P_1[1, 1], yi * P_1[2, 2] - P_1[1, 2],
                   yi * P_1[2, 3] - P_1[1, 3]]

            xi = x2[0, p]
            yi = x2[1, p]
            eq3 = [xi * P_2[2, 0] - P_2[0, 0], xi * P_2[2, 1] - P_2[0, 1], xi * P_2[2, 2] - P_2[0, 2],
                   xi * P_2[2, 3] - P_2[0, 3]]
            eq4 = [yi * P_2[2, 0] - P_2[1, 0], yi * P_2[2, 1] - P_2[1, 1], yi * P_2[2, 2] - P_2[1, 2],
                   yi * P_2[2, 3] - P_2[1, 3]]

            A = np.stack([eq1, eq2, eq3, eq4])

            u, s, vh = np.linalg.svd(A)

            p_3d = vh[3]
            p_3d /= p_3d[3]
            X_tri[0,p] = p_3d[0]
            X_tri[1,p] = p_3d[1]
            X_tri[2,p] = p_3d[2]

            # Tell if its in front of both cameras

            p_img1 = T_1@p_3d
            p_img2 = T_21@p_3d

            if(p_img1[2]>0 and p_img2[2]>0):
                score+=1

        if score>best_score:
            best_score = score
            best_P = P_2
            best_T = T_12
            best_tri = X_tri

    if plt:
        drawRefSystem(ax, T_1, '-', 'C1')
        drawRefSystem(ax, best_T, '-', 'C2')
        #names=['A','B','C','D']
        #for T,name in zip(T_12s,names):
        #        drawRefSystem(ax,T,'-',name)
        ax.scatter(best_tri[0, :], best_tri[1, :], best_tri[2, :], marker='.')
        plt.show()

        X_p1 = P_1@best_tri
        X_p1 /= X_p1[2,:]

        plt.figure(10)
        ax = plt.axes()
        ax.set_title('SfM, C1. GT in blue, triangulated points in red.')
        plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
        plt.plot(x1[0, :], x1[1, :], 'bx', markersize=8)
        plt.plot(X_p1[0, :], X_p1[1, :], 'rx', markersize=8)
        plt.draw()  # We update the figure display

        X_p2 = best_P @ best_tri
        X_p2 /= X_p2[2,:]

        plt.figure(11)
        ax = plt.axes()
        ax.set_title('SfM, C2. GT in blue, triangulated points in red.')
        plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
        plt.plot(x2[0, :], x2[1, :], 'bx', markersize=8)
        plt.plot(X_p2[0, :], X_p2[1, :], 'rx', markersize=10)
        plt.draw()  # We update the figure display

        plt.show()

    return best_tri, T_1, best_T

def approxH(p_1,p_2):

    M = np.zeros((p_1.shape[1]*2, 9))

    for i in range(p_1.shape[1]):
        x_0 = p_1[0, i]
        x_1 = p_2[0, i]
        y_0 = p_1[1, i]
        y_1 = p_2[1, i]

        M[2*i] = [x_0, y_0, 1, 0, 0, 0, -x_1*x_0, -x_1*y_0, -x_1]
        M[2*i+1] = [0, 0, 0, x_0, y_0, 1, -y_1*x_0, -y_1*y_0, -y_1]

    u, s, vh = np.linalg.svd(M)

    v = vh[-1]

    H = np.array([[v[0],v[1],v[2]],
                [v[3],v[4],v[5]],
                [v[6],v[7],v[8]]])
    H /= H[2,2]

    print("Approximated homography from matches:")
    print(H)

    return H

def showH(img1,img2,H_2_1,x1,H_title="H"):

    # Ensure x1 is in homogeneous 2D
    x1h = np.ones((3,x1.shape[1]))
    x1h[:2,:] = x1[:2,:]

    # Create an image with both images stacked vertically (img1 on top of img2)
    img12 = np.concatenate((img1, img2), axis=0)

    # Compute points in img2, then add img1's height to move them onto img2
    x2 = H_2_1@x1h
    x2 /= x2[2,:]
    x2[1,:] += img1.shape[0] # number of rows = height in pixels

    # Draw a line from every x1 point to its relocated x2
    fig, ax = plt.subplots()
    ax.set_title('Showing homography: '+H_title)
    for i in range(x1.shape[1]):
        xvalues = [x1[0,i],x2[0,i]]
        yvalues = [x1[1,i],x2[1,i]]
        plt.plot(xvalues,yvalues,'o',linestyle='--')
    plt.imshow(img12)
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # Load ground truth
    T_w_c1 = np.loadtxt('T_w_c1.txt')
    T_w_c2 = np.loadtxt('T_w_c2.txt')

    K_c = np.loadtxt('K_c.txt')
    X_w = np.loadtxt('X_w.txt')

    # Load points on each image
    # Shape: 2 rows, 103 columns. Each column is an (x,y) pair for a point.

    x1 = np.loadtxt('x1Data.txt')
    x2 = np.loadtxt('x2Data.txt')

    I_3_4 = [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0]]

    # Projection matrices for both cameras (from ground truth)
    P_1 = K_c@I_3_4@np.linalg.inv(T_w_c1)
    P_2 = K_c@I_3_4@np.linalg.inv(T_w_c2)

    print("Ground truth: C1 projection matrix")
    print(P_1)
    print("Ground truth: C2 projection matrix")
    print(P_2)

    # SVD triangulation

    # Prepare a matrix for storing the triangulated points, shape (4,103)
    X_w_tri = np.ones(X_w.shape)

    for p in range(x1.shape[1]):

        xi = x1[0,p]
        yi = x1[1,p]
        eq1 = [xi*P_1[2,0]-P_1[0,0],xi*P_1[2,1]-P_1[0,1],xi*P_1[2,2]-P_1[0,2],xi*P_1[2,3]-P_1[0,3]]
        eq2 = [yi*P_1[2,0]-P_1[1,0],yi*P_1[2,1]-P_1[1,1],yi*P_1[2,2]-P_1[1,2],yi*P_1[2,3]-P_1[1,3]]

        xi = x2[0,p]
        yi = x2[1,p]
        eq3 = [xi*P_2[2,0]-P_2[0,0],xi*P_2[2,1]-P_2[0,1],xi*P_2[2,2]-P_2[0,2],xi*P_2[2,3]-P_2[0,3]]
        eq4 = [yi*P_2[2,0]-P_2[1,0],yi*P_2[2,1]-P_2[1,1],yi*P_2[2,2]-P_2[1,2],yi*P_2[2,3]-P_2[1,3]]

        A = np.stack([eq1,eq2,eq3,eq4])

        u, s, vh = np.linalg.svd(A)

        p_3d = vh[3]/vh[3,3]
        X_w_tri[0,p] = p_3d[0]
        X_w_tri[1,p] = p_3d[1]
        X_w_tri[2,p] = p_3d[2]

    # Plot ground truth next to triangulated 3d points
    fig3D = plt.figure(1)

    # Plot legend
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_title('Ground truth (blue) and triangulated points (red)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot cameras and world frames
    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    # Plot ground truth
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')
    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], color='b', marker='.')
    # Plot triangulated points
    ax.scatter(X_w_tri[0, :], X_w_tri[1, :], X_w_tri[2, :], color='r', marker='.')
    plt.show()

    plt.close(1)

    img1 = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_BGR2RGB)

    # PART 2 - EPIPOLAR GEOMETRY

    # Epipolar lines

    F_21_test = np.loadtxt('F_21_test.txt')
    interactiveEpip(F_21_test)

    # Fundamental matrix

    F_true_12, E_true_12, T_true_12 = getFundamental(T_w_c1,T_w_c2,K_c,K_c)
    F_true_21, E_true_21, T_true_21 = getFundamental(T_w_c2,T_w_c1,K_c,K_c)
    interactiveEpip(F_true_21)

    F_approx_12 = approxFundamental(x1,x2)
    E_approx_12 = getEfromF(F_approx_12,K_c,K_c)
    F_approx_21 = approxFundamental(x2,x1)
    E_approx_21 = getEfromF(F_approx_21,K_c,K_c)

    interactiveEpip(F_approx_12)

    # Structure from motion

    X_tri_c1, T_c1_c1, T_c1_c2 = SfM(E_true_12,K_c,K_c,x1,x2,plot=True,img1=img1,img2=img2)
    #X_tri, T_c2_c2, T_c2_c1 = SfM(E_true_21,K_c,K_c,x2,x1,plot=True,img1=img2,img2=img1)

    # Reconstruction of the 3d points obtained from the C1 frame during SfM
    X_tri_w = T_w_c1@X_tri_c1
    X_gt_c1 = np.linalg.inv(T_w_c1)@X_w
    T_gt_c1_c2 = np.linalg.inv(T_w_c1)@T_w_c2

    # Code for showing ground truth points in C1 space
    plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_title('Ground truth transformed onto C1 space (notice loss of scale)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Plot ground truth poses
    drawRefSystem(ax, T_c1_c1, '-', 'C1')
    drawRefSystem(ax, T_c1_c2, '-', 'C2sfm')
    drawRefSystem(ax, T_gt_c1_c2, '-', 'C2gt')
    # Plot ground truth points
    ax.scatter(X_gt_c1[0, :], X_gt_c1[1, :], X_gt_c1[2, :], color='b', marker='.')
    # Plot triangulated points
    ax.scatter(X_tri_c1[0, :], X_tri_c1[1, :], X_tri_c1[2, :], color='r', marker='.')
    plt.show()

    # Sfm_scale is fixed to 1 (as the world is constructed around C1, with no units)
    # GT has another scale which we need to eliminate so points are comparable
    SfM_scale = np.linalg.norm(T_c1_c2[:3,3]) # This is 1
    GT_scale = np.linalg.norm(T_gt_c1_c2[:3,3]) # This is not

    T_gt_c1_c2_s = ensamble_T(T_gt_c1_c2[:3,:3],T_gt_c1_c2[:3,3]/GT_scale)
    X_gt_c1_s = X_gt_c1/GT_scale

    plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_title('Ground truth transformed onto C1 space (Corrected scale)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Plot ground truth poses
    drawRefSystem(ax, T_c1_c1, '-', 'C1')
    drawRefSystem(ax, T_c1_c2, '-', 'C2sfm')
    drawRefSystem(ax, T_gt_c1_c2_s, '-', 'C2gt')
    # Plot ground truth points
    ax.scatter(X_gt_c1_s[0, :], X_gt_c1_s[1, :], X_gt_c1_s[2, :], color='b', marker='.')
    # Plot triangulated points
    ax.scatter(X_tri_c1[0, :], X_tri_c1[1, :], X_tri_c1[2, :], color='r', marker='.')
    plt.show()

    # SfM Evaluation
    # Compute the average and median Euclidean distance (unitless) between the ground truth
    # and the points obtained. For reference, a value of 1 equals the distance between both cameras.

    xdist = X_gt_c1_s[0,:] - X_tri_c1[0,:]
    ydist = X_gt_c1_s[1,:] - X_tri_c1[1,:]
    zdist = X_gt_c1_s[2,:] - X_tri_c1[2,:]
    dist = np.sqrt(xdist*xdist+ydist*ydist+zdist*zdist)

    avg_dist = np.average(dist)
    med_dist = np.median(dist)

    print("SfM: Avg error:", avg_dist, ", Median error:", med_dist)

    # PART 3 - HOMOGRAPHY

    Pi_1 = np.loadtxt('Pi_1.txt')
    n = Pi_1[:3]
    n = np.expand_dims(n,axis=0).T
    norm = np.linalg.norm(n)
    n /= norm
    d = Pi_1[3]
    d /= norm

    T_c2_c1 = np.linalg.inv(T_w_c2)@T_w_c1
    R_c2_c1 = T_c2_c1[:3,:3]
    t_c2_c1 = T_c2_c1[:3,3]
    t_c2_c1 = np.expand_dims(t_c2_c1,axis=0).T

    A = R_c2_c1-(t_c2_c1@n.T)/d
    H_2_1_gt = K_c@A@np.linalg.inv(K_c)

    print("Plane induced homography:")
    print(H_2_1_gt)

    # Homography visualization

    x1_f = np.loadtxt('x1FloorData.txt')
    x2_f = np.loadtxt('x2FloorData.txt')

    showH(img1,img2,H_2_1_gt,x1_f,H_title='Ground truth')

    # Homography estimation

    H_2_1_app = approxH(x1_f,x2_f)

    showH(img1,img2,H_2_1_app,x1_f,H_title='Approx. from matches')

    # Homography evaluation:
    # Compute average and median Euclidean distance (in pixels) between the ground
    # truth points on the displaced image (img2) and the ones computed from the homography.

    x2_f_gth = H_2_1_gt @ x1_f
    x2_f_gth /= x2_f_gth[2,:]
    xdist = x2_f_gth[0,:] - x2_f[0,:]
    ydist = x2_f_gth[1,:] - x2_f[1,:]
    dist = np.sqrt(xdist*xdist+ydist*ydist)

    avg_dist = np.average(dist)
    med_dist = np.median(dist)

    print("GT Homography: Avg error:",avg_dist,"px, Median error:",med_dist,"px")

    x2_f_app = H_2_1_app @ x1_f
    x2_f_app /= x2_f_app[2,:]
    xdist = x2_f_app[0, :] - x2_f[0, :]
    ydist = x2_f_app[1, :] - x2_f[1, :]
    dist = np.sqrt(xdist * xdist + ydist * ydist)

    avg_dist = np.average(dist)
    med_dist = np.median(dist)

    print("Approx. Homography: Avg error:", avg_dist, "px, Median error:", med_dist,"px")


