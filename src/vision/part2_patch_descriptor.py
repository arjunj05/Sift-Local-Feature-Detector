#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    def helper(row, col):
        if row < 0 or row >= len(image_bw) or col < 0 or col >= len(image_bw[0]):
            return 0
        return image_bw[row][col]
    fvs = np.zeros((len(X), feature_width ** 2))
    for i in range(len(X)):
        patch = np.zeros((feature_width, feature_width)) # do we assume feature_width is even
        for k in range(feature_width):
            for j in range(feature_width):
                patch[j][k] = helper(Y[i] + k - (feature_width // 2 - 1),X[i] + j - (feature_width // 2 -1))
        fv = patch.flatten()
        if np.linalg.norm(fv):
            fv /= np.linalg.norm(fv)
        fvs[i, :] = fv  

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
