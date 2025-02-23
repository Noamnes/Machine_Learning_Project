import numpy as np
from scipy.ndimage import sobel

# write function that get two images and return the Move Erther Distance between them

def get_earth_mover_distance(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Computes the Move-Ether Distance between two grayscale images using
    Sobel gradients to extract features and a pixel-wise matching strategy.

    Parameters:
        image1 (np.ndarray): First input image (grayscale).
        image2 (np.ndarray): Second input image (grayscale).

    Returns:
        float: The computed Move-Ether Distance between the images.
    """
    
    # Ensure both images have the same shape
    assert image1.shape == image2.shape, "Images must have the same dimensions"
    
    # if the images are flattened, reshape them to 2D
    if len(image1.shape) == len(image2.shape) == 1:
        image1 = image1.reshape((int(np.sqrt(image1.shape[0])), int(np.sqrt(image1.shape[0]))))
        image2 = image2.reshape((int(np.sqrt(image2.shape[0])), int(np.sqrt(image2.shape[0]))))
    
    # Compute Sobel gradients to capture edge-based features
    grad_x1 = sobel(image1, axis=0, mode='constant')
    grad_y1 = sobel(image1, axis=1, mode='constant')
    
    grad_x2 = sobel(image2, axis=0, mode='constant')
    grad_y2 = sobel(image2, axis=1, mode='constant')

    # Compute gradient magnitude representations
    magnitude1 = np.sqrt(grad_x1**2 + grad_y1**2)
    magnitude2 = np.sqrt(grad_x2**2 + grad_y2**2)

    # Compute pixel-wise Euclidean distance
    distance = np.sum((magnitude1 - magnitude2) ** 2)
    
    return np.sqrt(distance)
