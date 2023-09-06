# External libraries
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

# Own libraries
from python.metadata.path import Path
from python.metadata.responses import Responses
from python.utils.image import convert_image
from python.utils.readers import read_images

if __name__ == "__main__":
    image = convert_image(Path.img_johan, Path.img_johan_transform)

    plt.imshow(image, cmap="gray")

    io.show()

    plt.close()

    list_cohort = read_images(Path.img_cohort)

    avg_cohort = sum(list_cohort) / len(list_cohort)

    plt.imshow(avg_cohort, cmap="gray")

    io.show()

    distance = np.linalg.norm(image - avg_cohort)

    print(f"Average of my face: {distance}")
    print(Responses.distance_face)
