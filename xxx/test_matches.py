import matplotlib.pyplot as plt
import numpy as np
import cv2


def euclidean_distance(actual, predicted):
    """ Computes the Mean of the translation error as euclidean distance"""
    return np.linalg.norm(actual - predicted)

def main():

    actual = np.array([[395, 292, 173, 419],
                       [20, 34, 62, 60],
                       [1, 1, 1, 1]])

    predicted = np.array([[168.88, 160.61, 215.32, 207.64],
                       [49.82, 65.51, 100.90, 112.47],
                       [1, 1, 1, 1]])


    distance = np.linalg.norm(actual[:,2] - predicted[:,1])

    print(distance)

if __name__ == '__main__':
    main()