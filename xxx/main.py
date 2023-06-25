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
    print('x1_SPSG= ', x1_SPSG)
    print('x2_SPSG= ', x2_SPSG)
    x1_SPSG = np.vstack((x1_SPSG.T, np.ones((1, x1_SPSG.shape[0]))))
    x2_SPSG = np.vstack((x2_SPSG.T, np.ones((1, x2_SPSG.shape[0]))))
    print('x1_SPSG= ', x1_SPSG)
    print('x2_SPSG= ', x2_SPSG)
    descs1_SPSG = match_data1['descriptors0'].T
    descs2_SPSG = match_data1['descriptors1'].T
    matches_SG = match_data1['matches']
    print('matches_SG.shape = ', matches_SG.shape)
    print('matches_SG= ', matches_SG)
    matches_SG_confidence = match_data1['match_confidence']


    print('x1_SPSG.shape= ', x1_SPSG.shape)
    print('x2_SPSG.shape', x2_SPSG.shape)
    print('matches_SG', matches_SG)
    print('matches_SG.shape', matches_SG.shape)
    np.savetxt('x1_SPSG.txt', x1_SPSG)
    np.savetxt('x2_SPSG.txt', x2_SPSG)
    np.savetxt('matches_SG.txt', matches_SG)

    '''
    count = 0
    for k in range(len(matches_SG)):
        if matches_SG[k] != -1:
            count += 1
    print('count = ', count)
    '''

    # Plot the matching points

    x1_matches_SG, x2_matches_SG = pld.plotMatchedPoints(matches_SG, x1_SPSG, x2_SPSG, new1, new2, 'Matched points between new 1 and new 2')

    # Quello che succede nella funzione precedente è che l'array matches comprende una serie di valori, quando
    # matches[i]= -1 , ciò significa che il keypoint in keypoint1 non ha corrispondenza in keypoints2.
    # Quindi, in sintesi, i valori numerici presenti nell'array matches rappresentano le corrispondenze tra i keypoints di due
    # insiemi di keypoints (keypoints1 e keypoints2)
    # e sono utilizzati per associare i punti corrispondenti nelle rispettive matrici x1_matches e x2_matches.
    # Quindi, quando all'interno di matches[i] ci sarà un valore != -1, ciò significa che ci sarà corrispondenza tra il
    # keypoints1 all'idice i e il corrispettivo keypoints2 che si trova al'indice matches[i]. Quindi il codice è ben fatto.

    print('x1_matches_SG_shape = ', x1_matches_SG.shape)
    print('x2_matches_SG_shape = ', x2_matches_SG.shape)
    np.savetxt('x1_matches_SG.txt', x1_matches_SG)
    np.savetxt('x2_matches_SG.txt', x2_matches_SG)


    # RANSAC

    x1_matches_rans, x2_matched_rans, matches_rans, dMatches_rans, p3Ds_rans, T_c2_c1_rans = rans.ransac(matches_SG, matches_SG_confidence, x1_SPSG, x2_SPSG, K, KC, descs1_SPSG, descs2_SPSG,  new1, new2)
    print(x1_matches_rans.shape)
    print(x2_matched_rans.shape)
    print(matches_rans.shape)

    np.savetxt('T_c2_c1_rans.txt', T_c2_c1_rans)
    np.savetxt('p3Ds_rans.txt', p3Ds_rans)
    np.savetxt('x1_matches_rans.txt', x1_matches_rans)
    np.savetxt('x2_matched_rans.txt', x2_matched_rans)
    np.savetxt('matchesssssss_rans.txt', matches_rans)


    count = 0
    for k in range(len(matches_rans)):
        if matches_rans[k] != -1:
            count += 1
    print('count = ', count)


    print('x1_matches_rans', x1_matches_rans)
    print('x1_matches_rans.shape', x1_matches_rans.shape)
    print('x2_matched_rans', x2_matched_rans)
    print('x2_matched_rans', x2_matched_rans.shape)
    print('matches_rans.shape =', matches_rans.shape)


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



    '''
    T_c2_c1_ba = np.loadtxt('T_c2_c1_ba.txt')
    p3Ds_ba = np.loadtxt('p3Ds_ba.txt')

    ax = pld.plotWorldFromC1(T_c2_c1_ba, p3Ds_ba, title="3D points from C1 after bundle adjustment")
    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()
    '''


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
    np.savetxt('xOld_SPSG', xOld_SPSG)
    np.savetxt('xNew_SPSG', xNew_SPSG)


    xNew_matches_SG, xOld_matches_SG = pld.plotMatchedPoints(matches_SG_newOld, xNew_SPSG, xOld_SPSG, new1, old,
                                                             'Matched points between picture 1 and old picture')

    np.savetxt('matches_SG_newOld.txt', matches_SG_newOld)

    print('xNew_SPSG.shape', xNew_SPSG.shape)
    print('xOld_SPSG.shape', xOld_SPSG.shape)
    print('matches_SG_newOld.shape', matches_SG_newOld.shape)
    print('xNew_matches_SG.shape', xNew_matches_SG.shape)
    np.savetxt('xNew_matches_SG', xNew_matches_SG)
    print('matches_SG_newOld.shape', matches_SG_newOld.shape)
    np.savetxt('xOld_matches_SG', xOld_matches_SG)
    print('xOld_matches_SG', xOld_matches_SG.shape)

    '''
    count = 0
    for k in range(len(matches_SG_newOld)):
        if matches_SG_newOld[k] != -1:
            count += 1
    print('count=', count)
    '''


    # Extract 2D and 3D points that can be used for recovering the pose

    points_2D_old = np.array([[], [], []])
    points_3D_old = np.array([[], [], [], []])
    points_2D_new = np.array([[], [], []])

    j = 0
    for i in range(len(matches_rans)):
        if (matches_rans[i] != -1):
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


    points_2D_old = np.loadtxt('points_2D_old.txt')
    points_3D_old = np.loadtxt('points_3D_old.txt')
    print('points_2d_old.shape = ', points_2D_old.shape)

    plt.figure()
    plt.imshow(old)
    plt.title('Points used for DLT')
    pld.plotMarkersImagePoints(points_2D_old, color='r', marker='+')
    plt.show()


    # DLT
    
    x2D = points_2D_old.T
    x3D = points_3D_old.T

    P_old, K_old, T_old = dlt.dlt(x2D, x3D)


    np.savetxt('P_old.txt', P_old)
    np.savetxt('K_old.txt', K_old)
    np.savetxt('T_old.txt', T_old)


    """
    P_old = np.loadtxt('P_old.txt')
    K_old = np.loadtxt('K_old.txt')
    T_old = np.loadtxt('T_old.txt')
    """


    # Check the residuals after DLT
    reconstruction2Dfull(old, K_old, C, T_old, points_3D_old, points_2D_old, title=f"Mean euclidian error old image")
    plt.title('Residual after DLT')
    plt.show()


    # Plot the 3D model and camera poses

    ax = pld.plotWorldFromC1_2(T_old, T_c2_c1_ba, p3Ds_ba, title="3D points and camera poses from C1")
    ax.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.show()


    # --------------------- Find changes ------------------------------


    # Manual removing of bad projected points

    #Viene calcolata la proiezione delle coordinate 3D (p3Ds_ba) sulla vecchia immagine (im3_projection_new) utilizzando la matrice di proiezione P_old.

    im3_projection_new = project(P_old, p3Ds_ba)
    im3_projection_new[:, -1] = im3_projection_new[:, -2]

    rows1, cols1 = np.shape(im3_projection_new)
    index1 = []

    for i in range(cols1 - 1):
        if im3_projection_new[0, i] < 0 or im3_projection_new[0, i] > 640 or im3_projection_new[1, i] < 0 or im3_projection_new[1, i] > 480:
            index1.append(i)

    im3_projection_new = np.delete(im3_projection_new, index1, axis=1)

    # Succede che Se un punto proiettato cade al di fuori dei limiti dell'immagine (0-640 coordinata X e 0-480 coordinata Y),verrà eliminato


    # Identify and plot the changes

    rows3, cols3 = np.shape(im3_projection_new)
    rows4, cols4 = np.shape(points_2D_old)

    simil = np.array([[], [], []])
    changes = np.array([[], [], []])


    # Parameters of the changes idetificator
    nb_pts_thresh = 2    #valore soglia (nb_pts_thresh) per il numero minimo di punti necessari per considerare un possibile cambiamento
    dist_thresh = 30     #soglia di distanza (dist_thresh) per considerare un punto come cambiamento


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




