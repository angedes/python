import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotData as pld
import fRANSAC
import hRANSAC
import lineRANSAC


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


    img1 = cv2.cvtColor(cv2.imread("new1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("new2.png"), cv2.COLOR_BGR2RGB)
    new1 = cv2.resize(img1, dsize=(640, 480))
    new2 = cv2.resize(img2, dsize=(640, 480))

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


    F = fRANSAC.fRANSAC_superglue()
    print('F = ', F)

    E = pld.getEfromF(F,K,K)
    print('E = ', E)




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
      
      '''
