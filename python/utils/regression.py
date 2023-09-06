# External libraries
import numpy as np
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def mnist_logistic_regression(normalization: bool = False) -> None:
    """Train a logistic regression classifier on the MNIST dataset to
     distinguish between 0s and 8s.

    Args:
        normalization: Whether to normalize pixel values to the range [0, 1].

    """
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

    logistic_regression = LogisticRegression(random_state=1234)
    logistic_regression.fit(train_data, train_labels)

    labels_predict = logistic_regression.predict(test_data)

    print(classification_report(test_labels, labels_predict))
