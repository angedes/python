#####################################################################################
#
# MRGCV Unizar - Computer vision - Course assignement
#
# Title: Course assignement get3DModel
#
# Date:
#
#####################################################################################
#
# Authors: Thomas HUYGHE
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import argparse
import matching
import homography as hm
import sfm
import sfm2
import plotData as pld
from common import project
from common import euclidean_error, euclidean_distance, euclidean_distance3D
from common import plotResidual
from bundleAdjustment import reconstruction2Dfull
import BA_and_PnP as BA_PnP
import bundleAdjustment as ba
import bundleAdjustmentNviews as ban
import dlt
import ransac as rans

im1 = cv2.cvtColor(cv2.imread("new1.png"), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread("new2.png"), cv2.COLOR_BGR2RGB)
im3 = cv2.cvtColor(cv2.imread("old.png"), cv2.COLOR_BGR2RGB)

resize = [640, 480]

new1 = cv2.resize(im1, dsize=resize)
new2 = cv2.resize(im2, dsize=resize)
old = cv2.resize(im3, dsize=resize)

K = np.loadtxt('K_mobilePhone.txt')
C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
        ])
KC = K @ C

def main():

    # ---------------------- Get the 3D model from matches -------------------------------------

    # Load the matches from SuperGlue

    match_data1 = np.load("new1_new2_matches.npz")

    x1_SPSG = match_data1['keypoints0']
    x2_SPSG = match_data1['keypoints1']
    x1_SPSG = np.vstack((x1_SPSG.T, np.ones((1, x1_SPSG.shape[0]))))
    x2_SPSG = np.vstack((x2_SPSG.T, np.ones((1, x2_SPSG.shape[0]))))
    descs1_SPSG = match_data1['descriptors0'].T
    descs2_SPSG = match_data1['descriptors1'].T
    matches_SG = match_data1['matches']
    matches_SG_confidence = match_data1['match_confidence']

    print('x1_SPSG.shape = ', x1_SPSG.shape)
    print('x1_SPSG.shape = ', x2_SPSG.shape)
    print('matches_SG.shape = ', matches_SG.shape)

    # Plot the matching points

    x1_matches_SG, x2_matches_SG = pld.plotMatchedPoints(matches_SG, x1_SPSG, x2_SPSG, new1, new2, 'Matched points between new 1 and new 2')

    print('x1_matches_SG.shape = ', x1_matches_SG.shape)
    print('x2_matches_SG.shape = ', x2_matches_SG.shape)



    # RANSAC
    
    x1_matches_rans, x2_matched_rans, matches_rans, dMatches_rans, p3Ds_rans, T_c2_c1_rans = rans.ransac(matches_SG, matches_SG_confidence, x1_SPSG, x2_SPSG, K, KC, descs1_SPSG, descs2_SPSG,  new1, new2)

    print('x1_matches_rans.shape = ', x1_matches_rans.shape)
    print('x2_matches_rans.shape = ', x2_matched_rans.shape)
    print('matches_rans.shape = ', matches_rans.shape)
    np.savetxt('RANS.txt', matches_rans)

    np.savetxt('T_c2_c1_rans.txt', T_c2_c1_rans)
    np.savetxt('p3Ds_rans.txt', p3Ds_rans)
    
    '''
    # Bundle adjustment 2 views

    print()
    print("-----BA from 2 views-----")
    # Optimisation by bundle adjustment and plotting of 3D and 2D points
    T_c2_c1_ba, p3Ds_ba = ba.optimisationByBA(K, T_c2_c1_rans, x1_matches_rans, x2_matched_rans, p3Ds_rans)

    np.savetxt('T_c2_c1_ba.txt', T_c2_c1_ba)
    np.savetxt('p3Ds_ba.txt', p3Ds_ba)
    
    
    ax = pld.plotWorldFromC1(T_c2_c1_ba, p3Ds_ba, title="3D points from C1 after bundle adjustment")
    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()
    """
    '''

    #"""
    T_c2_c1_ba = np.loadtxt('T_c2_c1_ba.txt')
    p3Ds_ba = np.loadtxt('p3Ds_ba.txt')
    
    ax = pld.plotWorldFromC1(T_c2_c1_ba, p3Ds_ba, title="3D points from C1 after bundle adjustment")
    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()
    #"""

    # 2D projection to check the optimized result

    P2_1 = KC @ T_c2_c1_ba

    x1_test = project(KC, p3Ds_ba)
    x2_test = project(P2_1, p3Ds_ba)

    fig = plt.figure()
    plt.title('3D points projection after BA')
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(new1)
    ax.set_title('Picture 1 projection')
    pld.plotMarkersImagePoints(x1_test, color='b', marker='+')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(new2)
    pld.plotMarkersImagePoints(x2_test, color='r', marker='+')
    ax.set_title('Picture 2 projection')
    plt.show()


    # ------------------- Get the old camera pose -------------------------------

    # Matches from Superglue
    match_data2 = np.load("new1_old_matches.npz")

    xNew_SPSG = match_data2['keypoints0']
    xOld_SPSG = match_data2['keypoints1']
    xNew_SPSG = np.vstack((xNew_SPSG.T, np.ones((1, xNew_SPSG.shape[0]))))
    xOld_SPSG = np.vstack((xOld_SPSG.T, np.ones((1, xOld_SPSG.shape[0]))))
    descsNew_SPSG = match_data2['descriptors0'].T
    descsOld_SPSG = match_data2['descriptors1'].T
    matches_SG_newOld = match_data2['matches']
    matches_SG_confidence_newOld = match_data2['match_confidence']

    np.savetxt('matches_SG_newOld.txt', matches_SG_newOld)
    np.savetxt('xOld_SPSG.txt', xOld_SPSG)
    print('xNew_SPSG.shape = ', xNew_SPSG.shape)
    print('xOld_SPSG.shape = ', xOld_SPSG.shape)

    xNew_matches_SG, xOld_matches_SG = pld.plotMatchedPoints(matches_SG_newOld, xNew_SPSG, xOld_SPSG, new1, old, 'Matched points between picture 1 and old picture')

    print('xNew_matches_SG.shape = ', xNew_matches_SG.shape)
    print('xOld_matches_SG.shape = ', xOld_matches_SG.shape)
    print('matches_SG_newOld.shape = ', matches_SG_newOld.shape)


    
    # Extract 2D and 3D points that can be used for recovering the pose
    
    matchesAfterRans = np.loadtxt('Matches_rans.txt')

    points_2D_old = np.array([[], [], []])
    points_3D_old = np.array([[], [], [], []])
    points_2D_new = np.array([[], [], []])

    j = 0
    for i in range(len(matchesAfterRans)):
        if (matchesAfterRans[i] != -1):
            if(matches_SG_newOld[i] != -1):
                point2D_1 = xNew_SPSG[:, i]
                point2D = xOld_SPSG[:, [matches_SG_newOld[i]]]
                point3D = p3Ds_ba[:, [j]]
                points_2D_new = np.hstack((points_2D_new, point2D_1.reshape(3, 1)))
                points_2D_old = np.hstack((points_2D_old, point2D))
                points_3D_old = np.hstack((points_3D_old, point3D))
            j = j + 1
    
    np.savetxt('points_2D_old.txt', points_2D_old)
    np.savetxt('points_3D_old.txt', points_3D_old)
    


    #"""
    points_2D_old = np.loadtxt('points_2D_old.txt')
    points_3D_old = np.loadtxt('points_3D_old.txt')
    #"""

    plt.figure()
    plt.imshow(old)
    plt.title('Points used for DLT')
    pld.plotMarkersImagePoints(points_2D_old, color='r', marker='+')
    plt.show()

    """
    # DLT
    
    x2D = points_2D_old.T
    x3D = points_3D_old.T

    P_old, K_old, T_old = dlt.dlt(x2D, x3D)


    np.savetxt('P_old.txt', P_old)
    np.savetxt('K_old.txt', K_old)
    np.savetxt('T_old.txt', T_old)
    """

    #"""
    P_old = np.loadtxt('P_old.txt')
    K_old = np.loadtxt('K_old.txt')
    T_old = np.loadtxt('T_old.txt')
    #"""

    """
    # Check the residuals after DLT
    reconstruction2Dfull(old, K_old, C, T_old, points_3D_old, points_2D_old, title=f"Mean euclidian error old image")
    plt.title('Residual after DLT')
    plt.show()
    """

    # Plot the 3D model and camera poses

    ax = pld.plotWorldFromC1_2(T_old, T_c2_c1_ba, p3Ds_ba, title="3D points and camera poses from C1")
    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()


    # --------------------- Find changes ------------------------------


    # Manual removing of bad projected points

    im3_projection_new = project(P_old, p3Ds_ba)
    im3_projection_new[:, -1] = im3_projection_new[:, -2]

    rows1, cols1 = np.shape(im3_projection_new)
    index1 = []

    for i in range(cols1 - 1):
        if im3_projection_new[0, i] < 0 or im3_projection_new[0, i] > 640 or im3_projection_new[1, i] < 0 or im3_projection_new[1, i] > 480:
            index1.append(i)

    im3_projection_new = np.delete(im3_projection_new, index1, axis=1)


    # Identify and plot the changes

    rows3, cols3 = np.shape(im3_projection_new)
    rows4, cols4 = np.shape(points_2D_old)

    simil = np.array([[], [], []])
    changes = np.array([[], [], []])


    # Parameters of the changes idetificator
    nb_pts_thresh = 2
    dist_thresh = 30


    for i in range(cols3 - 1):
        k = 0
        for j in range(cols4 - 1):
            if euclidean_distance(points_2D_old[:, j], im3_projection_new[:, i].reshape(3,1)) < dist_thresh :
                k = k+1
            else :
                k = k
        if k < nb_pts_thresh:
            changes = np.hstack((changes, im3_projection_new[:, i].reshape(3,1)))
        else:
            simil = np.hstack((simil, im3_projection_new[:, i].reshape(3, 1)))

    np.savetxt('changes.txt', changes)
    np.savetxt('simil.txt', simil)

    plt.imshow(old)
    plt.title('Changes (in red)')
    pld.plotMarkersImagePoints(changes, color='r', marker='+')
    pld.plotMarkersImagePoints(simil, color='b', marker='+')
    plt.show()
    


if __name__ == '__main__':
    main()




