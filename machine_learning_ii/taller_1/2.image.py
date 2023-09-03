# External libraries
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

# Own libraries
from machine_learning_ii.utils.image import convert_image
from machine_learning_ii.utils.readers import read_images


# Own libraries
from machine_learning_ii.metadata.path import Path


if __name__ == '__main__':

    image = convert_image(Path.img_johan)

    io.imsave(fname=Path.img_johan_transform, arr=image, plugin='pil')

    plt.imshow(image, cmap='gray')

    io.show()

    plt.close()

    list_cohort = read_images(Path.img_cohort)

    avg_cohort = sum(list_cohort) / len(list_cohort)

    plt.imshow(avg_cohort, cmap='gray')

    io.show()

    distance = np.linalg.norm(image, avg_cohort)

    print(f'Average of my face: {distance}')
