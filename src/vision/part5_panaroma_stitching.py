import numpy as np
import cv2 as cv

from vision.part1_harris_corner import get_harris_interest_points
from vision.part4_sift_descriptor import get_SIFT_descriptors
from vision.part3_feature_matching import match_features_ratio_test

def panorama_stitch(imageA, imageB):
    """
    ImageA and ImageB will be an image pair that you choose to stitch together
    to create your panorama. Feel free to play around with 
    different image pairs as a fun exercise!

    In this task, you are asked to use your feature detector pipeline you have implemented in project 2
    
    You will:
    - Detect interest points in the two images using your feature detector.
    - Match the interest points using feature matcher.
    - Use the matched points to compute the homography matrix using RANSAC.
    - Warp one of the images into the coordinate space of the other image 
      manually to create a stitched panorama (note: you may NOT use any 
      pre-existing warping function like `warpPerspective`).


    Please note that you can use the cv.findHomography function to compute the homography matrix that you will 
    need to stitch the panorama.
    

    Args:
        imageA: first image that we are looking at (from camera view 1) [A x B]
        imageB: second image that we are looking at (from camera view 2) [M x N]

    Returns:
        panorama: stitch of image 1 and image 2 using manual warp. Ideal dimensions
            are either:
            1. A or M x (B + N)
                    OR
            2. (A + M) x B or N)
    """
    result = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    imgA_grayscale = cv.cvtColor(imageA,cv.COLOR_BGR2GRAY)
    imgB_grayscale = cv.cvtColor(imageB,cv.COLOR_BGR2GRAY)

    A_x, A_y, A_c = get_harris_interest_points(imgA_grayscale)
    B_x, B_y, B_c = get_harris_interest_points(imgB_grayscale)

    imgA_descriptors = get_SIFT_descriptors(imgA_grayscale, A_x, A_y)
    imgB_descriptors = get_SIFT_descriptors(imgB_grayscale, B_x, B_y)

    matches, confidences = match_features_ratio_test(imgA_descriptors, imgB_descriptors)

    imgAPoints = np.array([ [A_x[i], A_y[i]] for (i, j) in matches])
    imgBPoints = np.array([ [B_x[j] + imageA.shape[1], B_y[j]] for (i, j) in matches])

    M, mask = cv.findHomography(imgAPoints, imgBPoints, cv.RANSAC)


    res_width, res_height = imageA.shape[1] + imageB.shape[1], max(imageA.shape[0], imageB.shape[0])
    result = np.zeros((res_height, res_width, 3))
    #result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    result[0:imageB.shape[0], imageA.shape[1]:imageA.shape[1] + imageB.shape[1]] = imageB  # Shift imageB to the right

    
    warped_image_A = cv.warpPerspective(imageA, M, (res_width, res_height))
    mask = (warped_image_A > 0)  # Create a mask of non-zero pixels
    result[mask] = warped_image_A[mask]  # Overwrite only valid pixel"""

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return result