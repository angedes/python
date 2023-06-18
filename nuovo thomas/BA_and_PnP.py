import matplotlib.pyplot as plt
import cv2

import bundleAdjustment as ba
import sfm
import bundleAdjustmentNviews as ban

from common import to_homogeneous, calculateTransformsDiff, euclidean_error, ensamble_T
import numpy as np
import plotData as pld

path_image_1 = 'new1.png'
path_image_2 = 'new2.png'
path_image_3 = 'new3.png'

def reconstruction2Dfull(image_pers_2, K, C, T_c2_c1, p3D_c1, x2Data, title="2D reconstruction"):    # Plot the 2D reconstruction
    x1, x2 = ba.reconstruction2D(K, C, T_c2_c1, p3D_c1)
    plt.figure()
    plt.imshow(image_pers_2)
    print(f"{title}")