import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from transformations import compute_tilt_angle

ArrayLike = Union[np.ndarray, pd.DataFrame]


def plot_image(image: ArrayLike, title: str = "Image", cmap: str = "gray") -> None:
    """
    Plots a 2D array as an image.

    Parameters:
    - image (2D array-like): The image data to display.
    - title (str): Title of the plot (default: "Image").
    - cmap (str): Colormap to use (default: "gray").
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.axis("off")  # Hide axes for better visualization
    plt.show()


def plot_grid(img_list: List[ArrayLike], col_count: int, figsize: Tuple[int, int] = (6, 4), titles_list: List[str]=None, bold: bool=False) -> None:
    """
    Plots a list of 2D array-like objects in a grid with num_col columns.
    
    Parameters:
    - img_list: list of 2D array-like (NumPy ndarray, Pandas DataFrame, etc.)
    - col_count: int, number of columns in the grid
    - figsize: tuple, size of the figure (width, height)
    - titles_list: list of str, optional, titles for each image
    - bold: bool, optional, whether to display titles in bold
    """
    num_items = len(img_list)
    num_row = (num_items + col_count - 1) // col_count  # Compute the required number of rows
    
    fig, axes = plt.subplots(num_row, col_count, figsize=figsize)
    axes = np.array(axes).reshape(num_row, col_count)  # Ensure axes is always a 2D array
    
    for idx, ax in enumerate(axes.flatten()):
        if idx < num_items:
            arr = img_list[idx]
            if isinstance(arr, pd.DataFrame):
                arr = arr.values  # Convert DataFrame to NumPy array if needed
            
            ax.imshow(arr, cmap='gray', aspect='auto')
            ax.axis('off')  # Hide axes
            
            if titles_list:
                title = titles_list[idx]
            else:
                title = f"Image {idx+1}"
            
            if bold:
                    title = f"**{title}**"
            
            ax.set_title(title, fontsize=10)
    
        else:
            ax.axis('off')  # Hide empty subplots
    
    plt.tight_layout()
    plt.show()


def plot_images_with_tilt_angles(images, col_count=5, figsize=(10, 6), bold=False):
    angles = [int(compute_tilt_angle(img)) for img in images]
    plot_grid(images, col_count, figsize=figsize, titles_list=[f'{angle}∠' for angle in angles], bold=bold)


def plot_before_after_transform(image_list, transformation, col_count=4, figsize=(5,5), **transform_kwargs):
    """
    Plots each image in image_list alongside its transformed version.

    Parameters
    ----------
    image_list : list of np.ndarray
        A list of images to be plotted before and after transformation.
    transformation : callable
        A function or callable object that takes an image (and additional kwargs)
        and returns a transformed image.
    col_count : int, optional
        Number of columns for the plot grid. Defaults to 4.
    figsize : tuple, optional
        Tuple specifying the (width, height) in inches for the figure. Defaults to (5, 5).
    **transform_kwargs : dict
        Arbitrary keyword arguments to be passed to the transformation function.

    Returns
    -------
    None
        Displays a grid of subplots with each original image and its transformed counterpart.
    """
    # Build a list: [orig1, transformed(orig1), orig2, transformed(orig2), ...]
    before_after_transform_list = [
        img for orig in image_list
        for img in (orig, transformation(orig, **transform_kwargs))
    ]
    # Assuming 'plot_grid' is a helper function that arranges images in a grid
    plot_grid(before_after_transform_list, col_count=col_count, figsize=figsize)
