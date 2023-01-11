import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import plotData as pld

from skimage.feature import match_descriptors
from skimage.measure import ransac



def main():
    match_data = np.load("new1_new2_matches.npz")

    x1_SPSG = match_data['keypoints0']
    x2_SPSG = match_data['keypoints1']
    x1_SPSG = np.vstack((x1_SPSG.T, np.ones((1, x1_SPSG.shape[0]))))
    x2_SPSG = np.vstack((x2_SPSG.T, np.ones((1, x2_SPSG.shape[0]))))
    # descs1_SPSG = match_data['descriptors0'].T
    # descs2_SPSG = match_data['descriptors1'].T

    print("Keypoints 0 : ", x1_SPSG)
    print("Keypoints 0 shape = : ", x1_SPSG.shape)
    print("Keypoints 1 : ", x2_SPSG)
    print("Keypoints 1 shape = : ", x2_SPSG.shape)
    # print("Descriptors 0 : ", descs1_SPSG)
    # print("Descriptors 1 : ", descs2_SPSG)

    matches_SG = match_data['matches']
    matches_SG_confidence = match_data['match_confidence']

    print("Matches : ", matches_SG)
    print("Matches confidence : ", matches_SG_confidence)




if __name__== '__main__':

    main()

    img1 = cv.cvtColor(cv.imread("new1.png"), cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(cv.imread("new2.png"), cv.COLOR_BGR2RGB)
    new1 = cv.resize(img1, dsize=(640, 480))
    new2 = cv.resize(img2, dsize=(640, 480))

    K = np.loadtxt('K_MyCamera.txt')
    I_3_4 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    KC = K @ I_3_4

    match_data = np.load("new1_new2_matches.npz")

    x1_SPSG = match_data['keypoints0']
    x2_SPSG = match_data['keypoints1']
    x1_SPSG = np.vstack((x1_SPSG.T, np.ones((1, x1_SPSG.shape[0]))))
    x2_SPSG = np.vstack((x2_SPSG.T, np.ones((1, x2_SPSG.shape[0]))))
    #descs1_SPSG = match_data['descriptors0'].T
    #descs2_SPSG = match_data['descriptors1'].T
    matches_SG = match_data['matches']
    matches_SG_confidence = match_data['match_confidence']

    len_matches = 0

    for i in range(len(matches_SG)):
        if matches_SG[i] != -1:
            len_matches += 1

    x1_matches_SG = np.empty([3, len_matches])
    x2_matches_SG = np.empty([3, len_matches])

    j = 0

    print('matches_SG.shape=', matches_SG.shape)
    print('len_matches = : ', len_matches)                           # len_matches = 261

    for i in range(len(matches_SG)):
        if matches_SG[i] != -1:
            x1_matches_SG[0, j] = x1_SPSG[0, i]
            x1_matches_SG[1, j] = x1_SPSG[1, i]
            x1_matches_SG[2, j] = 1

            x2_matches_SG[0, j] = x2_SPSG[0, matches_SG[i]]
            x2_matches_SG[1, j] = x2_SPSG[1, matches_SG[i]]
            x2_matches_SG[2, j] = 1

            j += 1


    print('x1_matches_SG: ', x1_matches_SG)
    print('x1_matches.shape = : ', x1_matches_SG.shape)
    print('x2_matches_SG: ', x2_matches_SG)
    print('x2_matches.shape = : ', x2_matches_SG.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(new1)
    ax.set_title('New1 picture')
    pld.plotMarkersImagePoints(x1_matches_SG, color='b', marker='+')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(new2)
    pld.plotMarkersImagePoints(x2_matches_SG, color='r', marker='+')
    ax.set_title('New2 picture')
    plt.show()


    def run_sfm(K0, K1, img1, img2 , X=None, Y=None, Z=None):

        R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        R_t_1 = np.empty((3, 4))
        P1 = np.matmul(K0, R_t_0)
        P2 = np.empty((3, 4))
        X = np.array([]) if X is None else X
        Y = np.array([]) if Y is None else Y
        Z = np.array([]) if Z is None else Z

        sift = cv.SIFT_create()
        kp0, desc0 = sift.detectAndCompute(img1, None)
        kp1, desc1 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc0,desc1,k=2)

        good = []
        pts1 = []
        pts2 = []

        # Lowe's paper

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append(m)
                pts1.append(kp0[m.queryIdx].pt)
                pts2.append(kp1[m.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

        # prendi inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        # operazione standard
        # da te K1 == K0, ma non mi ricordo se siano corrette come le ho messe
        E = np.matmul(np.matmul(np.transpose(K1), F), K0)
        retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K1)

        R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
        R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())

        # dovrebbe essere un'identita' la R_t_0

        print("R_t_0 \n" + str(R_t_0))
        print("R_t_1 \n" + str(R_t_1))

        P2 = np.matmul(K1, R_t_1)

        print("The projection matrix P1 \n" + str(P1))
        print("The projection matrix P2 \n" + str(P2))

        pts1 = np.transpose(pts1)
        pts2 = np.transpose(pts2)

        points_3d = cv.triangulatePoints(P1, P2, pts1, pts2)
        points_3d /= points_3d[3]

        X = np.concatenate((X, points_3d[0]))
        Y = np.concatenate((Y, points_3d[1]))
        Z = np.concatenate((Z, points_3d[2]))

        print("X 3D = ", X)
        print("Y 3D = ", Y)
        print("Z 3D = ", Z)


    # La funzione mi sputa fuori le matrici di rotazione, di proiezione e i punti triangolarizzati in 3D

    X = run_sfm(K, K, img1 , img2, X=None, Y=None, Z=None )
























'''
    F = fRANSAC.fRANSAC_superglue()
    print('F = ', F)
    E = pld.getEfromF(F,K,K)
    print('E = ', E)

    # SVD triangulation

    # But I need projection matrix for the triangularization
    # Prepare a matrix for storing the triangulated points, shape (4,103)
    X_w_tri = np.ones(x1_w.shape)

    for p in range(x1_matches_SG.shape[1]):
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

        p_3d = vh[3] / vh[3, 3]
        X_w_tri[0, p] = p_3d[0]
        X_w_tri[1, p] = p_3d[1]
        X_w_tri[2, p] = p_3d[2]

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

    img1 = cv2.cvtColor(cv2.imread("new1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("new2.png"), cv2.COLOR_BGR2RGB)


    # Ora vorrei applicare SfM (in plotData.py) in cui vado a fare svd di E
    # Structure from motion

    X_tri_c1, T_c1_c1, T_c1_c2 = pld.SfM(E, K, K, x1_matches_SG, x2_matches_SG, plot=True, img1=new1.png, img2=new2.png)

    F = fRANSAC.fRANSAC_superglue()
    E = K.T @ F @ K
    print('E = ', E)
    FROM HERE:
    # svd on E: to get U W V -> R,t are there (up to a sign) (see the link I sent you!)
    # find the combination that has most of points with depth > 0 (see the link I sent you!)
    #find correspondances of picture1 and old image -> need to go greyscale now! Do it same as you found between picture1 and picture2

    imagePoints = # points in old image
    objectPoints = # I will give code how to get from points in picture1 and intriniscs to 3d world coord (can look up on the internet)
    cameraMatrix = ? # still to see... :(
    # here to get the location of old camera give the above
    _, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
    Rt = cv2.Rodrigues(rvec)
    R = Rt.transpose()
    
    
    La tua lista di robe direi che vada bene, allora prendi solo le immagini "nuove", poi con l'esercizio 2 (direi) ti ricavi 
    l'intrinsics della tua camera, onestamente non so come se faccia. Poi con l'esercizio 3 ti ricavi la fundamental matrix. 
    Avendo K e F ti puoi ricavare l'essential matrix (dopo guardo come si fa, ma è semplice, devi risolvere un sistema). 
    Dalla matrix E puoi ricavarti le extrinsics delle tue immagini (una di queste sarà quella "reference", quindi sarà un'identità).
     Poi trovi le correspondances tra le tue immagini e l'immagine antica. Avendo tutte le K e R,t puoi ricavarti i punti in 3D dei
      punti matchati tra le tue immagini e quella antica. Avendo punti in 3D e le corrispondenti punti in 2D nel tuo image plane 
      della imagine antica, puoi ricavarti intrinsics ed extrinsics della tua camera antica. Gli extrinsics sono la location della 
      tua camera antica rispetto alla camera dell'immagine che hai preso come reference. La reference la prendi perché quando 
      ottieni la tua essential tra camera1 e camera2, in realtà tu ottieni la trasformazione tra reference camera1 e reference 
      camera2, però se setti implicitamente il world reference frame al reference camera2, ottieni la trasformazione camera1 to 
      world (cioè extrinsics), e la trasformazione camera2 to world si riduce ad un'identità.
      
      '''
