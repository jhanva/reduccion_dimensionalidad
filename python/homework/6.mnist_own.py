from unsupervised.python.dimensionality_reduction import SVD
from unsupervised.python.dimensionality_reduction import PCA
from unsupervised.python.dimensionality_reduction import TSNE

from python.utils.classification import mnist_logistic_regression

if __name__ == '__main__':
    svd = SVD(n_components=2)
    print('\nOwn SVD:')
    mnist_logistic_regression(dimensionality_reduction=svd)

    pca = PCA(n_components=2)
    print('\nOwn PCA:')
    mnist_logistic_regression(dimensionality_reduction=pca)

    tsne = TSNE(n_components=2)
    print('\nOwn t-SNE:')
    mnist_logistic_regression(dimensionality_reduction=tsne)
