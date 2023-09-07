# Own libraries
from python.utils.regression import mnist_logistic_regression

if __name__ == '__main__':
    # Model without processing
    mnist_logistic_regression()

    # Model with processing
    mnist_logistic_regression(True)
