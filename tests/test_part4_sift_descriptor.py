#!/usr/bin/python3

import copy
import pdb
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vision.part1_harris_corner import get_harris_interest_points
from vision.part3_feature_matching import match_features_ratio_test
from vision.part4_sift_descriptor import (
    get_magnitudes_and_orientations,
    get_gradient_histogram_vec_from_patch,
    get_SIFT_descriptors,
    get_feat_vec,
)
from vision.utils import load_image, evaluate_correspondence, rgb2gray, PIL_resize

ROOT = Path(__file__).resolve().parent.parent  # ../..


def test_get_magnitudes_and_orientations():
    """ Verify gradient magnitudes and orientations are computed correctly"""
    Ix = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Iy = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)

    # there are 3 vectors -- (1,0) at 0 deg, (0,1) at 90 deg, and (-1,1) and 135 deg
    expected_magnitudes = np.array([[1, 1, 1], [1, 1, 1], [np.sqrt(2), np.sqrt(2), np.sqrt(2)]])
    expected_orientations = np.array(
        [[0, 0, 0], [np.pi / 2, np.pi / 2, np.pi / 2], [3 * np.pi / 4, 3 * np.pi / 4, 3 * np.pi / 4]]
    )

    assert np.allclose(magnitudes, expected_magnitudes)
    assert np.allclose(orientations, expected_orientations)


def test_get_gradient_histogram_vec_from_patch():
    """ Check if weighted gradient histogram is computed correctly """
    window_magnitudes = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    A = 1/8 * np.pi # squarely in bin [0, pi/4]
    B = 3/8 * np.pi # squarely in bin [pi/4, pi/2]

    window_orientations = np.array(
        [
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]
    )

    wgh = get_gradient_histogram_vec_from_patch(window_magnitudes, window_orientations)

    expected_wgh = np.array(
        [
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.], # bin 4, magnitude 1
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], # bin 4, magnitude 0
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.], # bin 4
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], # bin 4
            [ 0.,  0.,  0.,  0.,  0., 32.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.], # bin 5
            [ 0.,  0.,  0.,  0.,  0., 32.,  0.,  0.], # bin 5, magnitude 2
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.], # bin 5
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0., 16.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0., 16.,  0.,  0.]
        ]
    ).reshape(128, 1)

    assert np.allclose(wgh, expected_wgh, atol=1e-1)


def test_get_feat_vec():
    """ Check if feature vector for a specific interest point is returned correctly """
    window_magnitudes = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    A = 1/8 * np.pi # squarely in bin [0, pi/4]
    B = 3/8 * np.pi # squarely in bin [pi/4, pi/2]
    C = 5/8 * np.pi # squarely in bin [pi/2, 3pi/4]

    window_orientations = np.array(
        [
            [ A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ],
            [ C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C ]
        ]
    )

    feature_width = 16

    x, y = 7, 8

    fv = get_feat_vec(x, y, window_magnitudes, window_orientations, feature_width)

    expected_fv = np.array(
        [
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.687, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.687, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.485, 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.   , 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ],
            [0. , 0. , 0. , 0. , 0. , 0.   , 0.485, 0. ]
        ]
    ).reshape(128, 1)

    assert np.allclose(fv, expected_fv, atol=1e-2)


def test_get_SIFT_descriptors():
    """ Check if the 128-d SIFT feature vector computed at each of the input points is returned correctly """

    image1 = np.array(
        [
            [75, 75, 14, 43, 98,  6, 40,  1, 25, 24, 15,  2, 98, 85, 31, 57, 83, 59, 49, 19],
            [21, 81, 21, 34, 53, 86, 97, 73, 27, 84,  8, 77, 61, 58, 23, 27, 37, 17, 11,  7],
            [26, 69, 30,  1, 51, 46, 26, 52, 23, 55, 31, 13, 80, 12, 53, 85, 95, 95, 39, 27],
            [30, 88, 28, 49, 11, 61, 61, 18,  8, 95, 41, 44, 86, 95, 13, 27, 61, 34, 27, 74],
            [50, 54, 56,  2,  1, 89, 29, 43, 14, 32, 71, 81, 35, 65, 72, 98, 10, 35, 19, 11],
            [81, 31, 99, 37, 12,  9, 42, 70, 70, 80, 53,  1, 28, 28, 98,  9, 46, 90, 97, 11],
            [83,  5,  5, 32,  7, 55, 37, 62, 58, 24, 25, 71, 72, 66, 87, 29, 93, 95, 73, 22],
            [25, 80, 58, 39, 75, 31, 51, 34, 70, 46, 64, 27, 48, 90, 48, 92, 83,  1, 85, 29],
            [11, 70, 19, 45, 85, 63, 13, 82, 37, 64, 35, 66, 45, 29, 96, 24, 66, 60, 97, 97],
            [94, 12, 22, 84, 78, 87, 20,  1, 46, 56, 25, 58, 85, 78, 41, 62, 20, 11, 66, 38],
            [38, 36, 50, 12, 12,  4, 96, 95, 39, 20, 98, 72, 32, 71, 97, 27, 80, 78, 72, 83],
            [55, 13, 33, 60, 99, 42, 38, 82, 39, 47, 45, 61, 52, 41, 36, 40, 83, 36, 60, 28],
            [70,  9, 21, 62, 39, 54, 62, 87, 78, 35, 49, 74, 74, 57, 82, 35, 52, 40,  5, 97],
            [33, 39, 72, 53, 14, 16, 34, 16, 34,  4, 11, 28, 20,  1, 25,  8, 24, 27,  9, 25],
            [24, 51, 98, 81, 75, 15, 38,  2,  1, 75, 32, 70,  0, 35, 84, 93, 22, 35, 20, 28],
            [ 2,  9, 81, 81, 14,  3,  3, 95,  1, 39, 42,  0, 58, 76, 84, 14, 33, 36, 18, 37],
            [31, 17, 82, 68, 32, 28, 94, 55, 41, 87, 87, 85, 72,  0, 64, 42, 31, 37,  5, 82],
            [27, 87, 16, 48, 10, 16, 12, 51, 37, 48, 90, 51, 91, 45, 50,  7, 33, 45, 70, 52],
            [68, 98, 22, 56, 97, 28,  3, 98, 16, 84, 70, 73, 12, 62, 96, 76,  1, 81, 63, 81],
            [57, 81, 37, 25, 71, 88, 66, 15, 45, 62, 17, 65, 93, 95, 70, 55, 54, 42, 99, 81],
        ]
    ).astype(np.float32)

    X1, Y1 = np.array([8, 9]), np.array([8, 9])

    SIFT_descriptors = get_SIFT_descriptors(image1, X1, Y1)

    expected_SIFT_descriptors = np.array([
        [
            [0.3027, 0.0000, 0.0000, 0.5002, 0.0887, 0.1180, 0.0000, 0.4655],
            [0.2380, 0.3154, 0.1913, 0.3534, 0.1726, 0.2776, 0.2465, 0.2905],
            [0.0000, 0.2258, 0.3024, 0.2979, 0.3552, 0.2729, 0.2563, 0.2632],
            [0.2540, 0.2442, 0.1864, 0.1524, 0.3069, 0.1348, 0.2772, 0.3738],
            [0.0000, 0.3472, 0.0992, 0.0000, 0.2689, 0.4017, 0.2354, 0.3303],
            [0.1269, 0.1546, 0.2495, 0.3119, 0.2087, 0.2562, 0.1670, 0.3255],
            [0.2043, 0.2543, 0.2108, 0.0000, 0.1540, 0.3572, 0.1628, 0.0000],
            [0.2685, 0.3619, 0.1750, 0.1207, 0.4541, 0.0000, 0.2035, 0.1657],
            [0.3014, 0.2412, 0.2909, 0.3749, 0.3041, 0.0000, 0.2097, 0.1696],
            [0.2509, 0.2014, 0.3242, 0.3083, 0.2040, 0.1745, 0.3382, 0.3107],
            [0.1582, 0.3309, 0.3120, 0.2585, 0.2654, 0.1857, 0.1581, 0.0000],
            [0.2341, 0.2527, 0.3731, 0.1605, 0.0000, 0.2409, 0.1350, 0.2237],
            [0.4964, 0.0000, 0.0000, 0.5121, 0.3089, 0.3274, 0.2288, 0.3985],
            [0.2206, 0.3569, 0.2569, 0.2215, 0.3833, 0.1438, 0.3636, 0.0000],
            [0.1118, 0.2347, 0.1641, 0.0000, 0.4021, 0.1861, 0.4827, 0.2163],
            [0.3231, 0.3042, 0.2044, 0.1870, 0.1757, 0.3769, 0.0000, 0.3306],
        ],
        [
            [0.1359, 0.1774, 0.1871, 0.4785, 0.2064, 0.0000, 0.0000, 0.5156],
            [0.2327, 0.3168, 0.0000, 0.3259, 0.2386, 0.2920, 0.2593, 0.1849],
            [0.1121, 0.2853, 0.3047, 0.2252, 0.2738, 0.2791, 0.2015, 0.3573],
            [0.2393, 0.3497, 0.0000, 0.1490, 0.3312, 0.3633, 0.2250, 0.3044],
            [0.0000, 0.2526, 0.2666, 0.2651, 0.2345, 0.4344, 0.1853, 0.3319],
            [0.2067, 0.1912, 0.2735, 0.2438, 0.1421, 0.1706, 0.2377, 0.0974],
            [0.0000, 0.1007, 0.2679, 0.0000, 0.3103, 0.3950, 0.0613, 0.0000],
            [0.1078, 0.3538, 0.2428, 0.2158, 0.2579, 0.0000, 0.2298, 0.2722],
            [0.1031, 0.2358, 0.3038, 0.2530, 0.2973, 0.2501, 0.3412, 0.2014],
            [0.2685, 0.3992, 0.2867, 0.3428, 0.2276, 0.0000, 0.2237, 0.3037],
            [0.1547, 0.3459, 0.4665, 0.2187, 0.2312, 0.0000, 0.1546, 0.0000],
            [0.3544, 0.3352, 0.2590, 0.1184, 0.1287, 0.2514, 0.0646, 0.1824],
            [0.5384, 0.0000, 0.0771, 0.3208, 0.2622, 0.1006, 0.0000, 0.3963],
            [0.1554, 0.0000, 0.0904, 0.4022, 0.4929, 0.0982, 0.3555, 0.0000],
            [0.1824, 0.2507, 0.0000, 0.1828, 0.0000, 0.3393, 0.4976, 0.2115],
            [0.3324, 0.2651, 0.0000, 0.0000, 0.1820, 0.3084, 0.2723, 0.3417],
        ]
    ]).reshape(2, 128)
    assert np.allclose(SIFT_descriptors, expected_SIFT_descriptors, atol=1e-1)


def test_feature_matching_speed():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must run in under 90 seconds.
    """
    start = time.time()
    image1 = load_image(f"{ROOT}/data/1a_notredame.jpg")
    image2 = load_image(f"{ROOT}/data/1b_notredame.jpg")
    eval_file = f"{ROOT}/ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(copy.deepcopy(image1_bw))
    X2, Y2, _ = get_harris_interest_points(copy.deepcopy(image2_bw))

    image1_features = get_SIFT_descriptors(image1_bw, X1, Y1)
    image2_features = get_SIFT_descriptors(image2_bw, X2, Y2)

    matches, confidences = match_features_ratio_test(image1_features, image2_features)
    print("{:d} matches from {:d} corners".format(len(matches), len(X1)))

    end = time.time()
    duration = end - start
    print(f"Your Feature matching pipeline takes {duration:.2f} seconds to run on Notre Dame")

    MAX_ALLOWED_TIME = 90  # sec
    assert duration < MAX_ALLOWED_TIME


def test_feature_matching_accuracy():
    """
    Test how long feature matching takes to execute on the Notre Dame pair.
    This unit test must achieve at least 80% accuracy.
    """
    image1 = load_image(f"{ROOT}/data/1a_notredame.jpg")
    image2 = load_image(f"{ROOT}/data/1b_notredame.jpg")
    eval_file = f"{ROOT}/ground_truth/notredame.pkl"
    scale_factor = 0.5
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))
    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))
    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    X1, Y1, _ = get_harris_interest_points(copy.deepcopy(image1_bw))
    X2, Y2, _ = get_harris_interest_points(copy.deepcopy(image2_bw))

    image1_features = get_SIFT_descriptors(image1_bw, X1, Y1)
    image2_features = get_SIFT_descriptors(image2_bw, X2, Y2)

    matches, confidences = match_features_ratio_test(image1_features, image2_features)

    acc, _ = evaluate_correspondence(
        image1,
        image2,
        eval_file,
        scale_factor,
        X1[matches[:, 0]],
        Y1[matches[:, 0]],
        X2[matches[:, 1]],
        Y2[matches[:, 1]],
    )

    print(f"Your Feature matching pipeline achieved {100 * acc:.2f}% accuracy to run on Notre Dame")

    MIN_ALLOWED_ACC = 0.80  # 80 percent
    assert acc > MIN_ALLOWED_ACC
