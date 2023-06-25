import numpy as np
import cv2
import matching
import sfm
import plotData as pld
import matplotlib.pyplot as plt


def ransac(matches_SG, matches_SG_confidence, x1_SPSG, x2_SPSG, K, KC, descs1_SPSG, descs2_SPSG, im1, im2):
    dmatches_SG = []
    for i in range(len(matches_SG)):
        if matches_SG[i] != -1:
            dmatches_SG.append(cv2.DMatch(_queryIdx=i, _trainIdx=matches_SG[i], _distance=( 1 -matches_SG_confidence[i])))

    x1_matches_rans, x2_matched_rans, matches_rans, dMatchesAfterF_rans, p3Ds_rans, T_c2_c1_rans = matchRANSAC(x1_SPSG, x2_SPSG, matches_SG, dmatches_SG, K, KC, descs1_SPSG, descs2_SPSG, im1, im2)

    return x1_matches_rans, x2_matched_rans, matches_rans, dMatchesAfterF_rans, p3Ds_rans, T_c2_c1_rans



def matchRANSAC(x1, x2, matchesList, dMatchesList, K, KC, descs1, descs2, im1, im2):
    matches = matchesList

    x1_matched = np.float32([x1[:, m.queryIdx] for m in dMatchesList]).T
    x2_matched = np.float32([x2[:, m.trainIdx] for m in dMatchesList]).T

    x1_kp = [cv2.KeyPoint(x1[0, i], x1[1,i], 4.0) for i in range(x1.shape[1])]
    x2_kp = [cv2.KeyPoint(x2[0, i], x2[1, i], 4.0) for i in range(x2.shape[1])]

    plt.figure()
    imgFMatched = cv2.drawMatches(im1, x1_kp, im2, x2_kp, dMatchesList,
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgFMatched, cmap='gray', vmin=0, vmax=255)
    plt.title("Raw SPSG matches")

    # Fundamental matrix
    print("- Fundamental matrix:")
    F_21, inliers = matching.computeFwithRANSAC(x1_matched, x2_matched)
    print(f"{np.sum(inliers)} inliers out of {x1_matched.shape[1]}")
    print(f"Inlier ratio: {np.sum(inliers)/x1_matched.shape[1]}")
    E_21 = K.T @ F_21 @ K
    T_c2_c1, p3Ds_c1 = sfm.reconstructCameras(E_21, KC, x1_matched[:, inliers], x2_matched[:, inliers])
    pld.plotWorldFromC1(np.linalg.inv(T_c2_c1), p3Ds_c1, "Cameras from F with RANSAC")
    dMatchesAfterF = [match for (match, inlier) in zip(dMatchesList, inliers.tolist()) if inlier]

    # Plot matches after Fundamental Matrix
    plt.figure()
    imgFMatched = cv2.drawMatches(im1, x1_kp, im2, x2_kp, dMatchesAfterF,
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgFMatched, cmap='gray', vmin=0, vmax=255)
    print(f"Matches after F with RANSAC: {len(dMatchesAfterF)}")
    plt.title("Matches after Fundamental Matrix with RANSAC")
    plt.show()

    j=0

    for i in range(len(matches)):
        if(matches[i] != -1):
            if(not inliers[j]):
                matches[i] = -1
            j = j+1

    return x1_matched[:, inliers], x2_matched[:, inliers], matches, dMatchesAfterF, p3Ds_c1, T_c2_c1