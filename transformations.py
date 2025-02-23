import numpy as np
from scipy.ndimage import affine_transform, center_of_mass
import matplotlib.pyplot as plt
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

def center_image(img: np.ndarray) -> np.ndarray:
    """
    Centers a 28x28 grayscale digit image.
    
    Args:
        img (np.ndarray): A 28x28 grayscale image array.
        
    Returns:
        np.ndarray: A centered 28x28 image array.
    """
    # Compute the center of mass
    cy, cx = center_of_mass(img)
    
    # Compute the translation needed to center the image
    translate_y = 14 - cy
    translate_x = 14 - cx
    translation_matrix = np.array([[1, 0, translate_x], [0, 1, translate_y]])
    
    # Apply the translation
    centered_img = affine_transform(img, translation_matrix, offset=0, order=1, mode='constant', cval=0)
    return centered_img


import scipy.ndimage


def fourier_transform_features(image: np.ndarray) -> np.ndarray:
    """
    Computes the Fourier Transform features of a 28x28 image.

    Parameters:
        image (np.ndarray): A 2D (grayscale) or 3D (RGB) numpy array representing the image.

    Returns:
        np.ndarray: The magnitude spectrum of the Fourier Transform.
    """
    if image.shape[:2] != (28, 28):
        print(image.shape[:2])
        raise ValueError("Input image must be 28x28 pixels.")

    if image.ndim == 3:
        # Process each channel separately for RGB images
        transformed_channels = [fourier_transform_features(image[..., i]) for i in range(image.shape[2])]
        return np.stack(transformed_channels, axis=-1)

    # Compute the 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    return magnitude_spectrum


def fourier_transform_features_arr(images: np.ndarray) -> np.ndarray:
    """
    Computes the Fourier Transform features for an array of 28x28 images.

    Parameters:
        images (np.ndarray): A 3D (grayscale) or 4D (RGB) numpy array representing multiple images.

    Returns:
        np.ndarray: An array of magnitude spectrums of the Fourier Transform.
    """
    output = [fourier_transform_features(p) for p in images]
    return np.array(output)
