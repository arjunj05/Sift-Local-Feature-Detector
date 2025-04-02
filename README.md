# CS 4476/6476 Project 2: SIFT Local Feature Matching

# 🧠 SIFT-Based Local Feature Matching & Panorama Stitching

## 📌 Project Overview
This project implements a **SIFT-inspired feature matching pipeline** for matching images taken from multiple viewpoints of the same scene. The pipeline includes:
- **Harris corner detection**
- **Local patch and SIFT-like descriptors**
- **Ratio-test-based feature matching**
- **Panorama stitching using estimated homographies**


## 🚀 Features Implemented

### 🧭 1. Harris Corner Detector
- Uses image gradients (Sobel filter) and second-moment matrix
- Computes Harris response score:  
  $$ R = \text{det}(A) - \alpha \cdot (\text{trace}(A))^2 $$
- Applies max-pooling for **non-maximum suppression (NMS)**

### 🔲 2. Normalized Patch Descriptor
- Extracts fixed-size grayscale patches centered at interest points
- Normalizes each patch to unit norm for simple but effective matching
- Expected matching accuracy on Notre Dame: ~40–50%

### 📌 3. Ratio Test Feature Matching
- Implements Lowe's **Nearest Neighbor Distance Ratio (NNDR)** test:
  $$ \text{Ratio} = \frac{||f_1 - f_{match1}||}{||f_1 - f_{match2}||} $$
- Filters robust matches by thresholding this ratio
- Returns top confident matches and visualizes them

### 🌀 4. SIFT-Like Descriptor
- Computes 4×4 grid of 8-bin gradient histograms over 16×16 patch
- Applies Gaussian weighting and orientation binning
- Performs **Square Root SIFT normalization** for robustness
- Achieves >80% accuracy on Notre Dame image pair

### 🌄 5. Panorama Stitching (Hand-Graded)
- Detects features and estimates **homography** using `cv2.findHomography()`
- Warps and blends images using `cv2.warpPerspective()`
- Stitching tested on 3 given panoramas + 1 custom pair



## Getting started

- See [Project 0](https://github.gatech.edu/cs4476/project-0) for detailed environment setup.
- Ensure that you are using the environment `cv_proj2`, which you can install using the install script `conda/install.sh`.


