import numpy as np

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


def convert_image(path_read: str, path_save: str = None) -> np.array:
    """

    Args:
        path_read:
        path_save:

    Returns:

    """
    image = io.imread(path_read)

    image = resize(image, (256, 256), anti_aliasing=True)

    gray_image = rgb2gray(image)

    if path_save:
        io.imsave(path_save, arr=image, plugin='pil')

    return gray_image
