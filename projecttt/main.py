import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotData as pld



from skimage.feature import match_descriptors
from skimage.measure import ransac
from mpl_toolkits.mplot3d import Axes3D


def get_pose(pts1, pts2, K1, K2, image_shape, img1=None, img2=None):
    print("GET POSE")
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K1, method=cv2.RANSAC, prob=0.99, threshold=3.0)
    pts1 = pts1[mask.astype(bool).squeeze()]
    pts2 = pts2[mask.astype(bool).squeeze()]
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K1)
    M1 = np.hstack((R, t))
    M2 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    pts1 = pts1[mask.astype(bool).squeeze()]
    pts2 = pts2[mask.astype(bool).squeeze()]
    P1 = np.dot(K1, M1)
    P2 = np.dot(K2, M2)
    points_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
    points_4d = points_4d_hom / points_4d_hom[-1:, :]
    points_3d = points_4d[:3, :].T
    points_3d = points_3d[points_3d[:, 2] > 0, :]

    pts1_back = np.matmul(P1, np.concatenate((points_3d.T, np.ones(shape=(1, points_3d.shape[0]))), axis=0))[:2, :].T
    pts2_back = np.matmul(P2, np.concatenate((points_3d.T, np.ones(shape=(1, points_3d.shape[0]))), axis=0))[:2, :].T

    for pts in pts1_back:
        cv2.circle(img1, (int(pts[1]), int(pts[0])), 10, (0, 255, 0), 3)
    for pts in pts1:
        cv2.circle(img1, (int(pts[1]), int(pts[0])), 10, (255, 0, 0), 3)
    cv2.imwrite("original_and_reproj1.png", img1)
    for pts in pts2_back:
        cv2.circle(img2, (int(pts[1]), int(pts[0])), 10, (0, 255, 0), 3)
    for pts in pts2:
        cv2.circle(img2, (int(pts[1]), int(pts[0])), 10, (255, 0, 0), 3)
    cv2.imwrite("original_and_reproj2.png", img2)
    val1, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera([points_3d.astype(np.float32)], [pts1.astype(np.float32)],
                                                          image_shape, K1, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    val2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera([points_3d.astype(np.float32)], [pts2.astype(np.float32)],
                                                          image_shape, K2, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    print(f"QUALITY: {val1:.2f} {val2:.2f}")

    return M1, M2


def get_correspondances(img1, img2):
    # Initiate SIFT detector
    print("SIFT")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=20)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    print("MATCH")
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imwrite("correspondances_all.png", img3)
    for pts in pts1:
        cv2.circle(img1, (int(pts[1]), int(pts[0])), 5, (0, 255, 0))
    cv2.imwrite("correspondances1.png", img1)
    for pts in pts2:
        cv2.circle(img2, (int(pts[1]), int(pts[0])), 5, (0, 255, 0))
    cv2.imwrite("correspondances2.png", img2)
    return pts1, pts2


def triangulate(K1, K2, M1, M2, pts1, pts2):
    P1 = K1 @ M1  # Cam1 is the origin
    P2 = K2 @ M2  # R, T from stereoCalibrate
    # points1 is a (N, 1, 2) float32 from cornerSubPix
    points1u = cv2.undistortPoints(pts1, cameraMatrix=K1, distCoeffs=None)
    points2u = cv2.undistortPoints(pts2, cameraMatrix=K2, distCoeffs=None)
    points4d = cv2.triangulatePoints(P1, P2, points1u, points2u)
    points3d = (points4d[:3, :] / points4d[3, :]).T
    print('points3d', points3d)
    return points3d


if __name__ == '__main__':


    K = np.loadtxt('K_MyCamera.txt')
    I_3_4 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    KC = K @ I_3_4

    img1 = cv2.cvtColor(cv2.imread('new2.png'), cv2.COLOR_BGR2RGB)  # queryImage
    img2 = cv2.cvtColor(cv2.imread('new1.png'), cv2.COLOR_BGR2RGB)  # trainImage

    # with the points from Sift

    pts1, pts2 = get_correspondances(img1.copy(), img2.copy())
    M1, M2 = get_pose(pts1, pts2, K, K, img1.shape[:2][::-1], img1.copy(), img2.copy())
    #print('pts1 = ', pts1)
    #print('pts1.shape = ', pts1.shape)
    #print('pts2 = ', pts2)
    #print('pts2.shape = ', pts2.shape)
    points3d = triangulate(K, K, M1, M2, pts1, pts2)
    #print('points3d.shape = ', points3d.shape)


    # with Superglue

    match_data = np.load("new2_new1_matches.npz")
    x1_SPSG = match_data['keypoints0']
    x2_SPSG = match_data['keypoints1']
    x1_SPSG = np.vstack((x1_SPSG.T, np.ones((1, x1_SPSG.shape[0]))))
    x2_SPSG = np.vstack((x2_SPSG.T, np.ones((1, x2_SPSG.shape[0]))))
    print("Keypoints 0 : ", x1_SPSG)
    print("Keypoints 1 : ", x2_SPSG)
    matches_SG = match_data['matches']
    matches_SG_confidence = match_data['match_confidence']
    print("Matches_SG : ", matches_SG)
    print("Matches_SG_shape : ", matches_SG.shape)
    img1 = cv2.cvtColor(cv2.imread("new1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("new2.png"), cv2.COLOR_BGR2RGB)
    new1 = cv2.resize(img1, dsize=(640, 480))
    new2 = cv2.resize(img2, dsize=(640, 480))

    len_matches = 0
    for i in range(len(matches_SG)):
        if matches_SG[i] != -1:
            len_matches += 1

    x1_matches_SG = np.empty([3, len_matches])
    x2_matches_SG = np.empty([3, len_matches])
    j = 0
    print('Matches_SG.shape=', matches_SG.shape)
    # print('len_matches = : ', len_matches)
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
    #print('x1_matches_SG.shape = : ', x1_matches_SG.shape)
    print('x2_matches_SG: ', x2_matches_SG)
    #print('x2_matches_SG.shape = : ', x2_matches_SG.shape)
    x1 = (x1_matches_SG[:2,:]).T
    x2 = (x2_matches_SG[:2, :]).T
    #print('x1 = ', x1)
    #print('x2 = ', x2)
    #print('x1.shape = ', x1.shape)
    #print('x2.shape = ', x2.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(new2)
    ax.set_title('New2 picture')
    pld.plotMarkersImagePoints(x1_matches_SG, color='b', marker='+')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(new1)
    pld.plotMarkersImagePoints(x2_matches_SG, color='r', marker='+')
    ax.set_title('New1 picture')
    plt.show()

    M1, M2 = get_pose(x1, x2, K, K, img1.shape[:2][::-1], img1.copy(), img2.copy())
    points3d = triangulate(K, K, M1, M2, x1, x2)
    #print('points3d.shape = ', points3d.shape)



















    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    '''