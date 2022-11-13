import numpy as np


def load_file(filename):
    data = np.loadtxt(filename)
    # each line in the file becomes an row in matrix
    X = data[:, :-1]  # take all the rows for all columns except the last one
    Y = data[:, -1]
    return X, Y


def logreg_inference(X, w, b):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    return p


if __name__ == "__main__":
    data = np.load("model.npz")
    w = data["arr_0"]
    b = data["arr_1"]

    X, Y = load_file("titanic-test.txt")
    P = logreg_inference(X, w, b)
    Yhat = (P > 0.5)
    accuracy = (Y == Yhat).mean()
    print("Test accuracy: ", 100 * accuracy)