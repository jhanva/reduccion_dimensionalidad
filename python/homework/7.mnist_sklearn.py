from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from python.utils.classification import mnist_logistic_regression

if __name__ == '__main__':
    svd = TruncatedSVD(n_components=2)
    print('\nScikit-learn SVD:')
    mnist_logistic_regression(dimensionality_reduction=svd)

    pca = PCA(n_components=2)
    print('\nScikit-learn PCA:')
    mnist_logistic_regression(dimensionality_reduction=pca)

    tsne = TSNE(n_components=2)
    print('\nScikit-learn t-SNE:')
    mnist_logistic_regression(dimensionality_reduction=tsne)
