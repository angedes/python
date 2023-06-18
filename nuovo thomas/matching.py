import numpy as np
from common import euclidean_error_pair
from math import log, floor
import random
import homography as hm
import sfm

import plotData as pld
from matplotlib import pyplot as plt

import cv2


def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        if dist[indexSort[0]] < minDist \
                and (len(indexSort)==1 or (len(indexSort)>1 and dist[indexSort[0]] < distRatio * dist[indexSort[1]])):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches


def computeMaxTries(inlier_ratio, success_rate, dof):
    return log(1 - success_rate) / log(1 - inlier_ratio**dof)

def computeHwithRANSAC(x1, x2, inlier_sigma=0.5, inlier_ratio=0.8, success_rate=0.99):
    """ Computes the Homography using RANSAC """

    maxTries = computeMaxTries(inlier_ratio, success_rate, 4)
    print(f"Initial max tries: {maxTries}")
    best_inliers = None
    best_votes = 0
    currIter = 0
    best_4_matches = None
    while currIter < maxTries:
        # Select 4 random points
        x_selection = random.sample(list(range(x1.shape[1])), 4)
        # Calculate homography
        x1_selected = x1[:, x_selection]
        x2_selected = x2[:, x_selection]
        H_21 = hm.computeH(x1_selected, x2_selected)
        # Project points in im2
        x2_projected = H_21 @ x1
        x2_projected /= x2_projected[2]
        # Calculate error of every point
        error = euclidean_error_pair(x2, x2_projected)
        # Check which are less than inlier_ratio
        inliers = error < inlier_sigma
        votes = np.sum(inliers)
        # If better change inliers
        if votes > best_votes:
            best_inliers = inliers
            best_votes = votes
            curr_inlier_ratio = best_votes / x1.shape[1]
            best_4_matches = x_selection
            if curr_inlier_ratio == 1.0:
                break
            if curr_inlier_ratio > inlier_ratio:
                inlier_ratio = curr_inlier_ratio
                maxTries = computeMaxTries(inlier_ratio, success_rate, 4)

        currIter += 1

    print(f"Computed iterations: {currIter}")
    print(f"Max tries after updating: {maxTries}")
    # Refit with all inliers for more accuracy
    bestH = hm.computeH(x1[:, best_inliers], x2[:, best_inliers])

    return bestH, best_inliers, best_4_matches


def computeFerror(F_21, x1, x2):
    # Project lines in im2
    l2 = F_21 @ x1
    # Calculate line-point errors
    normal_norm = np.sqrt(np.sum(l2[0:2, :] ** 2, axis=0))
    l2_norm = l2 / normal_norm
    # Compute error
    res = l2_norm.T @ x2
    res_abs = np.abs(res)
    return res_abs


def computeFwithRANSAC(x1, x2, inlier_sigma=2.5, inlier_ratio=0.5, success_rate=0.999):
    """ Computes the Fundamental Matrix using RANSAC """

    maxTries = computeMaxTries(inlier_ratio, success_rate, 8)
    print(f"Initial max tries: {maxTries}")

    best_inliers = None
    best_votes = -1
    currIter = 0

    while currIter < maxTries:
        # Select 8 random points
        x_selection = random.sample(list(range(x1.shape[1])), 8)

        # Calculate Fundamental matrix
        x1_selected = x1[:, x_selection]
        x2_selected = x2[:, x_selection]
        F_21 = sfm.computeF(x1_selected, x2_selected, normalize=True)

        res = np.diag(computeFerror(F_21, x1, x2))
        inliers = res < inlier_sigma

        votes = np.sum(inliers)
        if votes > best_votes:
            best_inliers = inliers
            best_votes = votes
            best_res = res
            best_F_21 = F_21
            curr_inlier_ratio = best_votes / x1.shape[1]
            if curr_inlier_ratio == 1.0:
                break
            # If better change inliers
            if curr_inlier_ratio > inlier_ratio:
                inlier_ratio = curr_inlier_ratio
                maxTries = computeMaxTries(inlier_ratio, success_rate, 8)

        currIter += 1

    print(f"Computed iterations: {currIter}")
    print(f"Max tries after updating: {maxTries}")
    # Refit with all inliers for more accuracy
    bestF = sfm.computeF(x1[:, best_inliers], x2[:, best_inliers], normalize=True)
    final_errors = computeFerror(bestF, x1, x2)
    best_inliers = final_errors < inlier_sigma

    final_errors = sorted(np.diag(final_errors), key = lambda x:float(x)) #Order list to calculate Median
    print(f"RANSAC errors:\n\tAverage: {sum(final_errors)/len(final_errors)}\n\tMedian: {final_errors[floor(len(final_errors)/2)]}")
    #print(f"RANSAC errors:\n\tAverage: {sum(final_errors) / len(final_errors)}")

    """
    if __debug__:
        im2 = cv2.cvtColor(cv2.imread("new2.png"), cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.title("RANSAC inliers and outliers")
        plt.imshow(im2)
        l2 = bestF @ x1
        im2h, im2w, _ = im2.shape
        for i in range(x1.shape[1]):
            color = 'b' if best_inliers[i][i] else 'r'
            pld.plot2Dline(l2[:, i], color=color, linewidth=1.0, im2w=im2w)
            plt.scatter(x2[0, i], x2[1, i], c=color)

        plt.xlim(0, im2w)
        plt.ylim(im2h, 0)
        plt.show(block=False)
    """

    U, s, V = np.linalg.svd(bestF)


    return bestF, np.diag(best_inliers)



def guided_matching(F_21, x1, descs1, x2, descs2, dMatches, r2n=0.5, dist=120.0, inlier_sigma=1.0):
    if __debug__:
        im2 = cv2.cvtColor(cv2.imread("new2.png"), cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.title("Epipolar lines from new matches")
        plt.imshow(im2)
        im2h, im2w, _ = im2.shape
    newMatches = []

    matched_src = set()
    matched_dst = set()
    for dm in dMatches:
        matched_src.add(dm.queryIdx)
        matched_dst.add(dm.trainIdx)
    res = computeFerror(F_21, x1, x2)
    candidates_all = res < inlier_sigma
    for i in range(x1.shape[1]):
        if i in matched_src:
            continue
        candidates = candidates_all[i, :]
        candidates_size = np.sum(candidates)
        if candidates_size == 0:
            continue
        desc_i_2d = descs1[i, :]
        desc_i_2d = np.expand_dims(desc_i_2d, axis=0)
        match = matchWith2NDRR(desc_i_2d, descs2[candidates, :], r2n, dist)
        if len(match) == 0:
            continue
        curr_dmatches = indexMatrixToMatchesList(match)
        match = curr_dmatches[0]
        candidates_index = candidates.nonzero()[0]
        local_train_index = match.trainIdx
        train_index = candidates_index[local_train_index]
        if train_index in matched_dst:
            continue
        newMatches.append(cv2.DMatch(_queryIdx=i, _trainIdx=train_index, _distance=match.distance))
        matched_src.add(i)
        matched_dst.add(train_index)
        if __debug__:
            l2 = F_21 @ x1[:, i]
            pld.plot2Dline(l2, color='r', linewidth=1.0, im2w=im2w)
            plt.scatter(x2[0, train_index], x2[1, train_index], c='b')
    if __debug__:
        plt.show(block=False)
        plt.xlim(0, im2w)
        plt.ylim(im2h, 0)
    return newMatches



