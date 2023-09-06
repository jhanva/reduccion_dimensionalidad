# External libraries
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


def convert_image(path_read: str, path_save: str = None) -> np.array:
    """Convert an image located at 'path_read' to grayscale, resize it to
     (256, 256) pixels, and optionally save it to 'path_save'.

    Args:
        path_read: The path to the input image file.
        path_save: The path to save the converted image.
         If not provided, the image won't be saved.

    Returns:
        np.array: The grayscale image as a NumPy array.

    """
    image = io.imread(path_read)

    image = resize(image, (256, 256), anti_aliasing=True)

    gray_image = rgb2gray(image)

    if path_save:
        io.imsave(path_save, arr=gray_image, plugin="pil")

    return gray_image
