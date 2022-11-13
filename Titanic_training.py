import numpy as np
import matplotlib.pyplot as plt


def logreg_inference(X, w, b):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    return p


def cross_entropy(P, Y):
    P = np.clip(P, 0.0001, 0.9999)  # forces the argument to stay in this range, to solve the numerical issue
    # (When the probability is very close to 1 or 0)
    return (-Y * np.log(P) - (1 - Y) * np.log(1 - P)).mean()


def logreg_train(X, Y, lambda_, lr=1e-3, steps=100000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    accuracies = []
    losses = []
    for step in range(steps):
        P = logreg_inference(X, w, b)
        if step % 100 == 0:
            Yhat = (P > 0.5)
            accuracy = (Yhat == Y).mean()
            accuracies.append(100 * accuracy)
            loss = cross_entropy(P, Y)
            losses.append(loss)
        grad_w = ((P - Y) @ X) / m + 2 * lambda_ * w
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, accuracies, losses


def load_file(filename):
    data = np.loadtxt(filename)
    # each line in the file becomes a row in matrix
    X = data[:, :-1]  # take all the rows for all columns except the last one
    Y = data[:, -1]
    return X, Y


if __name__ == "__main__":
    X, Y = load_file("titanic-train.txt")
    w, b, acc, losses = logreg_train(X, Y, 0.0, 0.005, 100000)

    plt.plot(acc)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy %")

    plt.figure()

    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("losses %")

    plt.show()

    # Question 1: to find my probability of surviving:
    X = np.array([1, 1, 22, 3, 2, 70])
    P = logreg_inference(X, w, b)
    print("My probability to survive: ", P * 100)

    # Question 2: to find the training accuracy of the trained model
    # Also used to compare with the acc in the evaluation method for deciding overfitting or underfitting
    print("Average accuracy: ", sum(acc) / acc.__len__())

    # save for the evaluation part
    np.savez("model.npz", w, b)

    # Question 3: how the individual feature influence the probability of surviving:
    print("Weights: ", w)
    # each weight is associated with the one of the 6 features

    # Question 5:
    # The 2 most influential features are the class and the sex:
    # the variables are categorical, the points were overlapped with one another, so some small random noise is added.
    X, Y = load_file("titanic-train.txt")
    Xrnd = X + np.random.randn(X.shape[0], X.shape[1]) / 20
    plt.scatter(Xrnd[:, 0], Xrnd[:, 1], c=Y)
    plt.xlabel("Class")
    plt.ylabel("Sex")
    plt.colorbar()
    plt.show()
