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

def getMotion(E,K1,K2,x1,x2,plot=False):

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

    I_3_4 = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0]])

    if plot:
        ##Plot the 3D cameras and the 3D points
        fig3D = plt.figure()

        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    """
    # WHY ISN'T IT EQUAL TO E ???
    t_matrix = np.array([[0,-t[2],t[1]],
                [t[2],0,-t[0]],
                [-t[1],t[0],0]])

    print(t_matrix@R_plus90)
    print((-1*t_matrix)@R_plus90)
    print((t_matrix)@R_minus90)
    print((-1*t_matrix)@R_minus90)
    print(E)
    """

    I3 = np.identity(3)
    T_1 = ensamble_T(I3,np.zeros((3,)))
    P_1 = K1@I_3_4@T_1
    T_2s = [ensamble_T(R_plus90,t),ensamble_T(R_plus90,-t),ensamble_T(R_minus90,t),ensamble_T(R_minus90,-t)]
    for T,i in zip(T_2s,range(4)):
        T_2s[i] = np.linalg.inv(T)
    P_2s = [K2@I_3_4@T_2s[0],K2@I_3_4@T_2s[1],K2@I_3_4@T_2s[2],K2@I_3_4@T_2s[3]]

    best_P = P_2s[0]
    best_T = T_2s[0]
    best_score = 0
    best_tri = None

    for T_2,P_2 in zip(T_2s,P_2s):

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
            p_img2 = T_2@p_3d

            if(p_img1[2]>0 and p_img2[2]>0):
                score+=1

        if score>best_score:
            best_score = score
            best_P = P_2
            best_T = T_2
            print(T_2)
            best_tri = X_tri

    if plt:
        drawRefSystem(ax, T_1, '-', 'C1')
        drawRefSystem(ax, T_2s[0], '-', 'C2')
        #names=['A','B','C','D']
        #for T,name in zip(T_2s,names):
        #        drawRefSystem(ax,T,'-',name)
        ax.scatter(best_tri[0, :], best_tri[1, :], best_tri[2, :], marker='.')
        # Matplotlib does not correctly manage the axis('equal')
        xFakeBoundingBox = np.linspace(0, 4, 2)
        yFakeBoundingBox = np.linspace(0, 4, 2)
        zFakeBoundingBox = np.linspace(0, 4, 2)
        plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
        plt.show()


        img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)

        X_p1 = P_1@best_tri
        X_p1 /= X_p1[2,:]

        plt.figure(10)
        plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
        plt.plot(x1[0, :], x1[1, :], 'bx', markersize=8)
        plt.plot(X_p1[0, :], X_p1[1, :], 'rx', markersize=8)
        plt.draw()  # We update the figure display

        X_p2 = best_P @ best_tri
        X_p2 /= X_p2[2,:]

        plt.figure(11)
        plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
        plt.plot(x2[0, :], x2[1, :], 'bx', markersize=8)
        plt.plot(X_p2[0, :], X_p2[1, :], 'rx', markersize=10)
        plt.draw()  # We update the figure display

        plt.show()


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

    print(H)

    return H

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

    # PART 2 - EPIPOLAR GEOMETRY

    # Epipolar lines

    F_21_test = np.loadtxt('F_21_test.txt')
    #interactiveEpip(F_21_test)

    # Fundamental matrix

    #F_true, E_true, T_true = getFundamental(T_w_c2,T_w_c1,K_c,K_c)
    F_true2, E_true2, T_true2 = getFundamental(T_w_c1,T_w_c2,K_c,K_c)

    F_approx = approxFundamental(x1,x2)

    # Essential matrix

    print(E_true2)
    getMotion(E_true2,K_c,K_c,x1,x2,plot=True)
    #getMotion(E_true2,K_c,K_c,x2,x1,plot=True)

    # PART 3 - HOMOGRAPHY

    Pi_1 = np.loadtxt('Pi_1.txt')

    """
    Rc2c1 = T_true[:3,:3]
    tc2c1 = T_true[:3,3]
    n = Pi_1[:3]
    d = Pi_1[3]

    H21 = K_c@(Rc2c1-tc2c1@n/d)@np.linalg.inv(K_c)
    H21 /= H21[2,2]
    point = np.array([250,200,1]).transpose()
    point2 = H21@point
    point2 /= point2[2]
    print(point)
    print(point2)
    print()
    """
    # Homography visualization

    # Homography estimation

    x1_f = np.loadtxt('x1FloorData.txt')
    x2_f = np.loadtxt('x2FloorData.txt')

    H = approxH(x1_f,x2_f)

    point = np.array([250,200,1]).transpose()
    point2 = H@point
    point2 /= point2[2]
    print(point)
    print(point2)



    """
    #plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)
    #plotNumbered3DPoints(ax, X_w_tri, 'b', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()
    plt.waitforbuttonpress()
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    """

    ## 2D plotting example
    """
    plt.figure(1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :],'rx', markersize=10)
    plotNumberedImagePoints(x1, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.figure(2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2[0,:4], x2[1,:4],'rx', markersize=10)
    #plotNumberedImagePoints(x2, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    drawEpipLine(F_approx, point1)
    drawEpipLine(F_approx, point2)
    drawEpipLine(F_approx, point3)
    drawEpipLine(F_approx, point4)
    plt.title('Image 2')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    """
