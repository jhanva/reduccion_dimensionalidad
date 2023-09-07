# External libraries
import matplotlib.pyplot as plt
from skimage import io
from unsupervised.python.dimensionality_reduction import SVD

# Own libraries
from python.metadata.path import Path

if __name__ == '__main__':
    picture = io.imread(Path.img_johan_transform)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for j, k in enumerate(range(2, 22, 2)):
        svd = SVD(n_components=k)
        matrix = svd.inverse_transform(picture)

        row = j // 5
        col = j % 5

        axes[row, col].imshow(matrix, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'n_components: {k}')

    plt.tight_layout()
    plt.show()
