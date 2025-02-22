import numpy as np
from scipy.ndimage import affine_transform

import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.stats import moment

def deskew_image(img: np.ndarray) -> np.ndarray:
    """
    Deskews a single 28x28 grayscale digit image using moment analysis.
    
    Args:
        img (np.ndarray): A 28x28 grayscale image array.
        
    Returns:
        np.ndarray: A deskewed 28x28 image array.
    """
    # Compute the image moments
    y, x = np.indices(img.shape)
    total_mass = img.sum()
    if total_mass == 0:
        return img  # Avoid division by zero

    x_center = (x * img).sum() / total_mass
    y_center = (y * img).sum() / total_mass

    # Compute second-order central moments
    mu_xx = ((x - x_center) ** 2 * img).sum() / total_mass
    mu_yy = ((y - y_center) ** 2 * img).sum() / total_mass
    mu_xy = ((x - x_center) * (y - y_center) * img).sum() / total_mass

    if mu_yy == 0:
        return img  # No skew detected

    skew = mu_xy / mu_yy  # Skew factor

    # Define the transformation matrix
    M = np.array([[1, skew, -skew * x_center], [0, 1, 0]])

    # Apply affine transformation
    deskewed_img = affine_transform(img, M, offset=0, order=1, mode='constant', cval=0)
    
    return deskewed_img

def compute_tilt_angle(image):
    # Normalize image to binary (thresholding)
    binary_img = (image > 128).astype(float)
    
    # Compute centroid
    cy, cx = center_of_mass(binary_img)

    # Compute second-order central moments
    y, x = np.indices(image.shape)
    mu20 = np.sum((x - cx)**2 * binary_img)
    mu02 = np.sum((y - cy)**2 * binary_img)
    mu11 = np.sum((x - cx) * (y - cy) * binary_img)
    
    # Compute orientation angle (in degrees)
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) * (180 / np.pi)
    return abs(theta)  # Absolute value of tilt angle