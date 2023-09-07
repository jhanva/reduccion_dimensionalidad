# Own libraries
from python.utils.classification import mnist_logistic_regression

if __name__ == '__main__':
    print('\nModel without processing')
    mnist_logistic_regression()

    print('\nModel with processing')
    mnist_logistic_regression(True)
