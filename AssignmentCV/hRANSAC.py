#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Line RANSAC fitting
#
# Date: 28 September 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import scipy.linalg as scAlg

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

    #print("Approximated homography from matches:")
    #print(H)

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

def hRANSAC_superglue(filename="image1_image2_matches.npz"):

    img1 = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_BGR2RGB)

    # Load matches
    npz = np.load(filename)

    # Descriptors located in destriptors0 and descriptors1, one column each, 256 rows
    # Keypoints located in keypoints0 and keypoints1, one row each, n columns for n keypoints

    kp0 = npz.f.keypoints0
    kp0h = np.ones((3,kp0.shape[0]))
    kp0h[:2,:] = kp0.T
    kp1 = npz.f.keypoints1
    kp1h = np.ones((3,kp1.shape[0]))
    kp1h[:2,:] = kp1.T
    matches = npz.f.matches
    matchlist = []
    kp0_matched = []

    for i in range(len(matches)):
        if matches[i]>-1:
            matchlist.append((i,matches[i]))
            kp0_matched.append(kp0h[:,i])
        # end if
    # end for
    kp0_matched = np.array(kp0_matched).T


    # parameters of random sample selection
    spFrac = 0.1  # spurious fraction
    P = 0.99999  # probability of selecting at least one sample without spurious
    pMinSet = 4  # number of points needed to compute the homography

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
    nAttempts = nAttempts.astype(int)
    print('nAttempts = ' + str(nAttempts))

    RANSACThreshold = 6
    nVotesMax = 0

    for kAttempt in range(nAttempts):

        nVotes = 0
        inliers = np.zeros(len(matchlist))

        # Compute the minimal set defining your model
        # For a homography, the minimum is 4 matches ko
        xSubSel = random.choices(matchlist, k=pMinSet)

        p1 = np.zeros((2,pMinSet))
        p2 = np.zeros((2,pMinSet))

        for m,i in zip(xSubSel,range(len(xSubSel))):
            p1[0,i] = kp0[m[0],0]
            p1[1,i] = kp0[m[0],1]
            p2[0,i] = kp1[m[1],0]
            p2[1,i] = kp1[m[1],1]

        H = approxH(p1,p2)
        kp1H = H @ kp0h
        kp1H/=kp1H[2]

        for m,i in zip(matchlist,range(len(matchlist))):

            kp1_gt = kp1h[:,m[1]]
            kp1_h = kp1H[:,m[0]]
            dist = np.sqrt(np.sum((kp1_gt-kp1_h)**2))
            if dist < RANSACThreshold:
                nVotes+=1
                inliers[i] = 1

        if nVotes > nVotesMax:
            print("New best")
            nVotesMax = nVotes
            print(nVotes,"votes")
            print(H)
            showH(img1,img2,H,kp0_matched[:,inliers==1])

def hRANSAC_sift(x1,x2):

    img1 = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_BGR2RGB)

    kp0h = x1
    kp1h = x2

    # parameters of random sample selection
    spFrac = 0.2  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious
    pMinSet = 4  # number of points needed to compute the homography

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
    nAttempts = nAttempts.astype(int)
    print('nAttempts = ' + str(nAttempts))

    RANSACThreshold = 5
    nVotesMax = 0

    for kAttempt in range(nAttempts):

        nVotes = 0
        inliers = np.zeros(x1.shape[1])

        # Compute the minimal set defining your model
        # For a homography, the minimum is 4 matches ko
        xSubSel = random.choices(range(x1.shape[1]), k=pMinSet)

        p1 = np.zeros((2,pMinSet))
        p2 = np.zeros((2,pMinSet))

        for m,i in zip(xSubSel,range(len(xSubSel))):
            p1[0,i] = kp0h[0,m]
            p1[1,i] = kp0h[1,m]
            p2[0,i] = kp1h[0,m]
            p2[1,i] = kp1h[1,m]

        H = approxH(p1,p2)
        kp1H = H @ kp0h
        kp1H/=kp1H[2]

        for i in range(x1.shape[1]):
            kp1_gt = kp1h[:,i]
            kp1_h = kp1H[:,i]
            dist = np.sqrt(np.sum((kp1_gt-kp1_h)**2))
            if dist < RANSACThreshold:
                nVotes+=1
                inliers[i] = 1

        if nVotes > nVotesMax:
            print("New best")
            nVotesMax = nVotes
            print(nVotes,"votes")
            print(H)
            showH(img1,img2,H,x1[:,inliers==1])

if __name__ == '__main__':

    hRANSAC_superglue()