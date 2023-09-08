import matplotlib.pyplot as plt
from keras.datasets import cifar10

from python.utils.rpca import RobustPCA

if __name__ == '__main__':
    (x_train, y_train), _ = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(-1, 32 * 32)

    x_train = x_train[: 100]

    robust_pca = RobustPCA()
    L, S, examples = robust_pca.pcp(x_train.T)

    example_images = x_train[:10]

    example_images = example_images.reshape(-1, 32, 32)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                   'frog', 'horse', 'ship', 'truck']

    fig, axs = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        axs[0, i].imshow(example_images[i])
        axs[0, i].axis('off')
        axs[0, i].set_title(class_names[y_train[i][0]])

    reconstructed_images = L.T.reshape(-1, 32, 32)

    for i in range(10):
        axs[1, i].imshow(reconstructed_images[i])
        axs[1, i].axis('off')
        axs[1, i].set_title('Filtered')

    plt.show()
