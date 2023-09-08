# External libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import warnings

from keras.datasets import mnist
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Own libraries
from python.metadata.path import Path

warnings.simplefilter("ignore")


def mnist_logistic_regression(
    normalization: bool = False,
    dimensionality_reduction=None,
    plot=False,
    save_model=False,
) -> None:
    """Train a logistic regression classifier on the MNIST dataset with
        optional dimensionality reduction.

    Args:
        normalization: Whether to normalize pixel values to the range [0, 1].
        dimensionality_reduction: An optional dimensionality reduction model
            (e.g., PCA, TruncatedSVD, t-SNE) to apply before training the
            classifier. If None, no dimensionality reduction is applied.
        plot:
        save_model:

    """
    # Load MNIST dataset
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    if normalization:
        train_data = train_data.astype(float) / 255.0
        test_data = test_data.astype(float) / 255.0

    train_data = train_data.reshape(-1, 28 * 28)
    test_data = test_data.reshape(-1, 28 * 28)

    train_filter = np.where((train_labels == 0) | (train_labels == 8))
    test_filter = np.where((test_labels == 0) | (test_labels == 8))

    train_data = train_data[train_filter]
    train_labels = train_labels[train_filter]

    test_data = test_data[test_filter]
    test_labels = test_labels[test_filter]

    # Apply dimensionality reduction if specified
    if dimensionality_reduction:
        train_data = dimensionality_reduction.fit_transform(train_data)
        test_data = dimensionality_reduction.fit_transform(test_data)

    if plot:
        x = train_data[:, 0]
        y = train_data[:, 1]

        plt.scatter(x, y, c=train_labels, cmap='viridis')

        plt.show()

    # Train logistic regression classifier
    logistic_regression = LogisticRegression(random_state=1234)
    logistic_regression.fit(train_data, train_labels)

    if save_model:
        joblib.dump(logistic_regression, Path.model)

    # Make predictions and print classification report
    labels_predict = logistic_regression.predict(test_data)
    print(metrics.accuracy_score(test_labels, labels_predict))
