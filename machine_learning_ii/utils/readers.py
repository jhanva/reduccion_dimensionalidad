# External libraries
import os

from skimage import io

# Own libraries
from machine_learning_ii.metadata.path import Path
from machine_learning_ii.utils.image import convert_image


def read_images(path: str) -> list:
    """Read and return a list of images from a specified directory.

    Args:
        path: The path to the directory containing the images.

    Returns:
        A list of image data.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        Exception: If an error occurs while reading an image.

    """
    images_list = os.listdir(path)

    images = []
    for path_image in images_list:
        read_path = os.path.join(Path.img_cohort, path_image)
        try:
            img = convert_image(read_path)
        except Exception as e:
            img = io.imread(read_path)

        images.append(img)

    return images
