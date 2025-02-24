import numpy as np
from scipy.ndimage import affine_transform, center_of_mass, gaussian_filter, map_coordinates
import scipy.ndimage
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
    mu20 = np.sum((x - cx) ** 2 * binary_img)
    mu02 = np.sum((y - cy) ** 2 * binary_img)
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


def random_elastic_deformation(
        image: np.ndarray,
        alpha: float = 34,
        sigma: float = 4,
        random_state: np.random.RandomState = None
) -> np.ndarray:
    """
    Apply a random elastic deformation to a 2D image.

    This function uses the approach of generating random displacement fields
    (one for the x-direction and one for the y-direction), smoothing them with
    a Gaussian filter, and then warping the original image according to these
    displacement fields.

    Parameters
    ----------
    image : np.ndarray
        The input 2D image (e.g. a NumPy array of shape (H, W)).
    alpha : float, optional
        Scales the magnitude of the displacement field. Larger values produce
        more pronounced distortions. Default is 34.
    sigma : float, optional
        Standard deviation for the Gaussian kernel used to smooth the random
        displacement fields. Default is 4.
    random_state : np.random.RandomState, optional
        A NumPy RandomState object for reproducibility. If None, a new RandomState
        is created using the current system time. Default is None.

    Returns
    -------
    np.ndarray
        The deformed image, having the same shape as the input image.

    Notes
    -----
    - This function assumes a single-channel 2D image. For multi-channel images,
      you can apply the same displacement fields to each channel separately.
    - The parameters `alpha` and `sigma` often require tuning depending on the
      desired extent of deformation and image resolution.
    - Inspired by the data augmentation technique described by Simard et al.
      in their work on distortions for handwritten character recognition.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    # Ensure image is 2D
    if image.ndim != 2:
        raise ValueError("random_elastic_deformation expects a 2D array.")

    # Generate random displacement fields
    h, w = image.shape
    dx = random_state.rand(h, w) * 2 - 1  # random in [-1, 1]
    dy = random_state.rand(h, w) * 2 - 1  # random in [-1, 1]

    # Smooth the displacement fields with a Gaussian filter
    dx = gaussian_filter(dx, sigma, mode="reflect") * alpha
    dy = gaussian_filter(dy, sigma, mode="reflect") * alpha

    # Create a coordinate mesh
    x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Distort the coordinates with dx, dy
    # map_coordinates expects (row_coords, col_coords)
    indices = np.array([
        x + dx,
        y + dy
    ])

    # Warp the input image based on the distorted grid
    deformed_image = map_coordinates(
        image,
        indices,
        order=1,  # bilinear interpolation
        mode='reflect'
    )

    return deformed_image


def flatten_images(images):
    return np.array([image.flatten() for image in images])


import numpy as np
import cv2
from scipy.special import jacobi
import tensorflow as tf  # Using TensorFlow for MNIST, you could use PyTorch instead

# Calculate radial polynomial for Zernike Moments
from math import factorial  # Import factorial from math module


def radial_poly(r, n, m):
    if (n - abs(m)) % 2 != 0 or abs(m) > n:
        return np.zeros_like(r)

    radial = np.zeros_like(r)
    for k in range((n - abs(m)) // 2 + 1):
        coef = ((-1) ** k * factorial(n - k)) / (
                factorial(k) *
                factorial((n + abs(m)) // 2 - k) *
                factorial((n - abs(m)) // 2 - k)
        )
        radial += coef * r ** (n - 2 * k)
    return radial


# Compute Zernike Moment for a single image
def zernike_moment(img, n, m):
    # Ensure image is 28x28 (MNIST size)
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Create coordinate grid
    x = np.arange(28) - 13.5  # Center at (13.5, 13.5)
    y = np.arange(28) - 13.5
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)

    # Normalize radius to [0,1]
    R = R / 13.5  # 13.5 is half the image size

    # Calculate Zernike polynomial
    V = radial_poly(R, n, m) * np.exp(1j * m * Theta)

    # Compute moment
    moment = np.sum(img * np.conj(V)) * (n + 1) / np.pi

    return moment


# Preprocess MNIST images with Zernike Moments
def preprocess_with_zernike(images, max_order=5):
    n_samples = images.shape[0]
    features = []

    for i in range(n_samples):
        img = images[i]
        moments = []

        # Calculate moments up to specified order
        for n in range(max_order + 1):
            for m in range(-n, n + 1, 2):  # m must satisfy (n-|m|) even
                moment = zernike_moment(img, n, m)
                # Store magnitude of complex moment
                moments.append(np.abs(moment))

        features.append(moments)

    return np.array(features)
