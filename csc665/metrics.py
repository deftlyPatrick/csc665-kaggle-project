import numpy as np


def mse(y_predicted, y_true):

    # return mean squared error
    return ((y_predicted - y_true) ** 2).mean()


def rmse(y_predicted, y_true):

    # return root mean squared error
    return np.sqrt(mse(y_predicted, y_true))


def rsq(y_predicted, y_true):

    # return score
    v = ((y_true - y_true.mean()) ** 2).mean()
    return 1 - mse(y_predicted, y_true) / v


def print_scores(model, x_train, x_test, y_train, y_test):

    # print mse and score values for provided arguments
    train_mse = mse(model.predict(x_train), y_train)
    test_mse = mse(model.predict(x_test), y_test)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    print('[', train_mse, test_mse, train_score, test_score, ']')
