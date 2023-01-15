import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import plotData as pld

from skimage.feature import match_descriptors
from skimage.measure import ransac
from mpl_toolkits.mplot3d import Axes3D


def run_sfm(K1, K2, img1, img2, X=None, Y=None, Z=None):
    # formula is P1 = K1@R_t_1; P2 = K2@R_t_2   with our case K1 =K2

    R_t_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    R_t_2 = np.empty((3, 4))
    P1 = K1 @ R_t_1
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
    matches = flann.knnMatch(desc0, desc1, k=2)

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
    # da te K2 == K1
    E = np.matmul(np.matmul(np.transpose(K2), F), K1)
    retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K2)

    R_t_2[:3, :3] = np.matmul(R, R_t_1[:3, :3])
    R_t_2[:3, 3] = R_t_1[:3, 3] + np.matmul(R_t_1[:3, :3], t.ravel())


    print("R_t_1 \n" + str(R_t_1))
    print("R_t_2 \n" + str(R_t_2))

    P2 = np.matmul(K2, R_t_2)

    print("The projection matrix P1 \n" + str(P1))
    print("The projection matrix P2 \n" + str(P2))

    pts1 = np.transpose(pts1)
    pts2 = np.transpose(pts2)

    points_3d = cv.triangulatePoints(P1, P2, pts1, pts2)
    points_3d /= points_3d[3]
    print('Points 3d with SIFT = ', points_3d)

    X = np.concatenate((X, points_3d[0]))
    Y = np.concatenate((Y, points_3d[1]))
    Z = np.concatenate((Z, points_3d[2]))

    # 3D points in X, Y, Z coordinates

    # print("X 3D = ", X)
    # print("Y 3D = ", Y)
    # print("Z 3D = ", Z)

    return R_t_1, R_t_2, P1, P2, X, Y, Z, points_3d



def calc_rep_error(P1, P2, K, R_t_1, R_t_2, imagePoint1, imagePoint2):

    imagePoint1, imagePoint2 = imagePoint1[:2, :], imagePoint2[:2, :]
    R_t_1 = R_t_1.astype(np.float64)
    #print('imagePoint1.shape', imagePoint1.shape)
    #print('imagePoint2.shape', imagePoint2.shape)
    #print('P1.shape', P1.shape)
    #print('P2.shape', P2.shape)

    # Triangulate

    point3D = cv.triangulatePoints(P1, P2, imagePoint1, imagePoint2).T
    #print('points 3D =', point3D)
    point3D = point3D[:, :3] / point3D[:, 3:4]
    print('Points 3D with Superglue = ', point3D)

    # We can filter 3D points based on if the z component it is negative and so we selected only with z > 0
    #point3D = point3D[point3D[:, 2] > 0,:]
    #print('points 3D with Superglue filtered =', point3D)

    # Reproject back into the two cameras
    R1, t1 = R_t_1[:, :3], R_t_1[:, 3]
    R2, t2 = R_t_2[:, :3], R_t_2[:, 3]
    t1 = np.matmul(R1.T, t1)
    t2 = np.matmul(R2.T, t2)

    #print('R1.dtype', R1.dtype)
    #print('R_t_2.shape',  R_t_2.shape)

    #print('R_t_2', R_t_2)

    rvec1, _ = cv.Rodrigues(R1)
    rvec2, _ = cv.Rodrigues(R2)

    p1, _ = cv.projectPoints(point3D, rvec1, t1, K, distCoeffs=None)
    p2, _ = cv.projectPoints(point3D, rvec2, t2, K, distCoeffs=None)

    print('p1.shape =', p1.shape)
    print('p2.shape =', p2.shape)

    # measure difference between original image point and reprojected image point

    reprojection_error1 = np.linalg.norm(imagePoint1 - p1[:,0,:].T, axis = 0)
    reprojection_error2 = np.linalg.norm(imagePoint2 - p2[:,0,:].T, axis = 0)

    print('reprojection_error1 = ', reprojection_error1)
    print('reprojection_error2 = ', reprojection_error2)

    # It is important to have the reprojection error that is low



if __name__== '__main__':

    # matching between new1 and the new2 images

    match_data = np.load("new1_new2_matches.npz")

    x1_SPSG = match_data['keypoints0']
    x2_SPSG = match_data['keypoints1']
    x1_SPSG = np.vstack((x1_SPSG.T, np.ones((1, x1_SPSG.shape[0]))))
    x2_SPSG = np.vstack((x2_SPSG.T, np.ones((1, x2_SPSG.shape[0]))))


    print("Keypoints 0 : ", x1_SPSG)
    print("Keypoints 0 shape = : ", x1_SPSG.shape)
    print("Keypoints 1 : ", x2_SPSG)
    print("Keypoints 1 shape = : ", x2_SPSG.shape)

    matches_SG = match_data['matches']
    matches_SG_confidence = match_data['match_confidence']

    print("Matches : ", matches_SG)
    print("Matches confidence : ", matches_SG_confidence)

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


    len_matches = 0

    for i in range(len(matches_SG)):
        if matches_SG[i] != -1:
            len_matches += 1

    x1_matches_SG = np.empty([3, len_matches])
    x2_matches_SG = np.empty([3, len_matches])

    j = 0

    #print('matches_SG.shape=', matches_SG.shape)
    #print('len_matches = : ', len_matches)

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
    print('x1_matches_SG.shape = : ', x1_matches_SG.shape)
    print('x2_matches_SG: ', x2_matches_SG)
    print('x2_matches_SG.shape = : ', x2_matches_SG.shape)

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


    # La funzione mi sputa fuori le matrici di rotazione, di proiezione e i punti triangolarizzati in 3D

    R_t_1, R_t_2, P1, P2, X, Y, Z , points_3d = run_sfm(K, K, img1 , img2, X=None, Y=None, Z=None )


    w = calc_rep_error(P1, P2, K, R_t_1, R_t_2, x1_matches_SG, x2_matches_SG)




























    '''
    # matching between new2 and the old images

    match_data_34 = np.load("new_old_matches.npz")

    x3_SPSG = match_data_34['keypoints0']
    x4_SPSG = match_data_34['keypoints1']
    x3_SPSG = np.vstack((x3_SPSG.T, np.ones((1, x3_SPSG.shape[0]))))
    x4_SPSG = np.vstack((x4_SPSG.T, np.ones((1, x4_SPSG.shape[0]))))
    # descs3_SPSG = match_data['descriptors0'].T
    # descs4_SPSG = match_data['descriptors1'].T

    print("Keypoints 0 : ", x3_SPSG)
    print("Keypoints 0 shape = : ", x3_SPSG.shape)
    print("Keypoints 1 : ", x4_SPSG)
    print("Keypoints 1 shape = : ", x4_SPSG.shape)

    matches_SG_34 = match_data_34['matches']
    matches_SG_confidence_34 = match_data_34['match_confidence']

    print("Matches_34 : ", matches_SG_34)
    print("Matches confidence_34 : ", matches_SG_confidence_34)

    img3 = cv.cvtColor(cv.imread("new.png"), cv.COLOR_BGR2RGB)
    img4 = cv.cvtColor(cv.imread("old.png"), cv.COLOR_BGR2RGB)
    new3 = cv.resize(img3, dsize=(640, 480))
    old = cv.resize(img4, dsize=(640, 480))


    len_matches_34 = 0

    for i in range(len(matches_SG_34)):
        if matches_SG_34[i] != -1:
            len_matches_34 += 1

    x3_matches_SG = np.empty([3, len_matches_34])
    x4_matches_SG = np.empty([3, len_matches_34])

    j = 0

    print('matches_SG_34.shape=', matches_SG_34.shape)
    print('len_matches = : ', len_matches)

    for i in range(len(matches_SG_34)):
        if matches_SG_34[i] != -1:
            x3_matches_SG[0, j] = x3_SPSG[0, i]
            x3_matches_SG[1, j] = x3_SPSG[1, i]
            x3_matches_SG[2, j] = 1

            x4_matches_SG[0, j] = x4_SPSG[0, matches_SG_34[i]]
            x4_matches_SG[1, j] = x4_SPSG[1, matches_SG_34[i]]
            x4_matches_SG[2, j] = 1

            j += 1

    print('x3_matches_SG: ', x3_matches_SG)
    print('x3_matches_SG.shape = : ', x3_matches_SG.shape)
    print('x4_matches_SG: ', x4_matches_SG)
    print('x4_matches_SG.shape = : ', x4_matches_SG.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(new3)
    ax.set_title('New picture')
    pld.plotMarkersImagePoints(x3_matches_SG, color='b', marker='+')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(old)
    pld.plotMarkersImagePoints(x4_matches_SG, color='r', marker='+')
    ax.set_title('Old picture')
    plt.show()

    '''








































































    '''
   

   
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
      
      
      
      
      Nella procudera hai bisogno di R_t per calcolarti R e t (ti servono per fare la projection), R_t (sia per 1 e 2, cioe' 
      R_t_1 e R_t_2) e' gia' calcolata in SfM, quindi li prendi da li e li sbatti in questa nuova funzione, cioe' la signature 
      della funzione e' : def calc_rep_error(P1, P2, K, R_t_1, R_t_2, imagePoint1, imagePoint2).
      Devi aver P1 e P2 shape [3,4] e i points sovrebbero essere [2,N] (con N uguali in entrambi i points).
      Per il resto se hai cose strane, guarda le shape se tornano. Se l'errore e' molto grande, prima prova a printare 
      R2, t2 in calc_rep_error e in R,t in run_sfm. Dovrebbero essere identiche le R e le t.
     
    '''

    '''
    A = np.eye(3, 4)
    pts = np.zeros([2, 10])
    points3D = cv.triangulatePoints(A, A, pts, pts)
    print(points3D.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points_3d[0]
    y = points_3d[1]
    z = points_3d[2]

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(x, y, z)
    plt.show()
    '''
