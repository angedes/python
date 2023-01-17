from plotData import *
from scipy.linalg import expm, logm
import math
import scipy.optimize as scOptim
from plotGroundTruth import plotResidual

import warnings

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    warnings.simplefilter("ignore", np.ComplexWarning)

    # Get the camera frame transformations
    T_w_c1 = np.loadtxt("T_wc1.txt")
    T_w_c2 = np.loadtxt("T_wc2.txt")
    T_c1_c2 = np.linalg.inv(T_w_c1)@T_w_c2
    T_c2_c1 = np.linalg.inv(T_w_c2)@T_w_c1

    # Read the images
    path_cam_1A = 'fisheye1_frameA.png'
    path_cam_1B = 'fisheye1_frameB.png'
    path_cam_2A = 'fisheye2_frameA.png'
    path_cam_2B = 'fisheye2_frameB.png'
    image_1A = cv2.imread(path_cam_1A)
    image_1B = cv2.imread(path_cam_1B)
    image_2A = cv2.imread(path_cam_2A)
    image_2B = cv2.imread(path_cam_2B)

    # Calibration
    K_1 = np.loadtxt('K_1.txt')
    K_2 = np.loadtxt('K_2.txt')
    D_1k = np.loadtxt('D1_k_array.txt')
    D_2k = np.loadtxt('D2_k_array.txt')
    d1k = np.concatenate(([1],D_1k[:4]))
    d2k = np.concatenate(([1],D_2k[:4]))

    # Load points on each image
    # Shape: 3 rows, n columns. Each column is an (x,y,w) for a point.
    x1 = np.loadtxt('x1.txt')
    x2 = np.loadtxt('x2.txt')
    x3 = np.loadtxt('x3.txt')
    x4 = np.loadtxt('x4.txt')

    # ----------------------------
    # PROJECTION MODEL
    # ----------------------------

    # Test points
    Test_X1 = np.array([3,2,10,1]).T
    Test_X2 = np.array([-5,6,7,1]).T
    Test_X3 = np.array([1,5,14,1]).T
    Test_X_c1 = np.stack([Test_X1,Test_X2,Test_X3],axis=1)
    Test_X_c2 = T_c2_c1 @ Test_X_c1

    def projectKannala(X,dk,K):

        phi = np.arctan2(X[1],X[0])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        theta = np.arctan2(np.sqrt(X[0]*X[0]+X[1]*X[1]),X[2])
        theta2 = theta*theta
        theta3 = theta*theta2
        theta5 = theta3*theta2
        theta7 = theta5*theta2
        theta9 = theta7*theta2

        d = np.array([theta, theta3, theta5, theta7, theta9]).T
        #print('d =', d)

        d_theta = d @ dk.T
        #print('d_theta =', d_theta)

        T = np.array([d_theta*cosphi,d_theta*sinphi,np.ones(X.shape[1])])
        u = K@T

        return u

    u1 = projectKannala(Test_X_c1,d1k,K_1)
    u2 = projectKannala(Test_X_c2,d2k,K_2)

    print('Test projection vectors u1 =\n', u1)
    #print('Test projection vectors u2 =\n', u2)


    # ----------------------------
    # UNPROJECTION MODEL
    # ----------------------------

    def unprojectKannala(u,dk,K):

        Xc = np.linalg.inv(K) @ u
        r = np.sqrt((Xc[0] * Xc[0] + Xc[1] * Xc[1]) / (Xc[2] * Xc[2]))

        phi = np.arctan2(Xc[1],Xc[0])
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

        num_points = u.shape[1]

        polyn = np.zeros((num_points,10))
        polyn[:,0] = dk[4]
        polyn[:,2] = dk[3]
        polyn[:,4] = dk[2]
        polyn[:,6] = dk[1]
        polyn[:,8] = dk[0]
        polyn[:,9] = -r.T

        theta = np.zeros(num_points)

        for i in range(num_points):

            roots = np.roots(polyn[i])
            theta[i] = roots[np.isreal(roots)][0]

        sinth = np.sin(theta)
        costh = np.cos(theta)

        v = np.array([sinth*cosphi,sinth*sinphi,costh])

        return v

    v1 = unprojectKannala(u1, d1k, K_1)
    v2 = unprojectKannala(u2, d2k, K_2)

    print('Test unprojection vectors v1 =\n', v1)
    #print('Test unprojection vectors v2 =\n', v2)

    # ----------------------------
    # PLANE TRIANGULATION
    # ----------------------------

    def planeTriangulation(v1s,v2s,T_c1_c2):

        num_points = v1s.shape[1]
        points = np.ones((4,num_points))

        for i in range(num_points):

            v1 = v1s[:,i]
            v2 = v2s[:,i]

            pi_sym1 = np.array([[-v1[1],v1[0],0,0]]).T
            pi_orto1 = np.array([[-v1[2]*v1[0],-v1[2]*v1[1],v1[0]*v1[0]+v1[1]*v1[1],0]]).T
            pi_sym2 = np.array([[-v2[1],v2[0],0,0]]).T
            pi_orto2 = np.array([[-v2[2]*v2[0],-v2[2]*v2[1],v2[0]*v2[0]+v2[1]*v2[1],0]]).T

            pi_sym12 = T_c1_c2.T@pi_sym1
            pi_orto12 = T_c1_c2.T@pi_orto1

            A = np.array([pi_sym12.T,pi_orto12.T,pi_sym2.T,pi_orto2.T]).squeeze()

            u, s, vh = np.linalg.svd(A)

            p_3d = vh[3] / vh[3, 3]
            points[:,i] = p_3d

        return points

    points_c2 = planeTriangulation(v1,v2,T_c1_c2)
    points_c1 = T_c1_c2 @ points_c2

    print("Recovered points")
    print(points_c1)

    # ----------------------
    # IMAGE EXAMPLE
    # ----------------------

    v1 = unprojectKannala(x1,d1k,K_1)
    v2 = unprojectKannala(x2,d2k,K_2)

    points_c2 = planeTriangulation(v1,v2,T_c1_c2)
    points_w = T_w_c2 @ points_c2

    # Plot the 2D points
    plt.figure(1)
    plt.imshow(image_1A, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :], 'b.')
    plt.title('Image 1')
    plt.draw()

    plt.figure(2)
    plt.imshow(image_1A, cmap='gray', vmin=0, vmax=255)
    plt.plot(x2[0, :], x2[1, :], 'b.')
    plt.title('Image 1')
    plt.draw()

    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')
    ax.scatter(points_w[0, :], points_w[1, :], points_w[2, :], marker='.')
    plt.title('Triangulated points')
    plt.show()

    