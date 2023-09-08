# Own libraries
from unsupervised.python.dimensionality_reduction import PCA, SVD, TSNE

from python.utils.classification import mnist_logistic_regression

if __name__ == '__main__':
    svd = SVD(n_components=2)
    print('\nOwn SVD:')
    mnist_logistic_regression(
        dimensionality_reduction=None, plot=True, save_model=True
    )

    pca = PCA(n_components=2)
    print('\nOwn PCA:')
    mnist_logistic_regression(dimensionality_reduction=pca, plot=True)

    tsne = TSNE(n_components=2, n_iter=10, target_perplexity=1.0)
    print('\nOwn t-SNE:')
    mnist_logistic_regression(dimensionality_reduction=tsne, plot=True)
