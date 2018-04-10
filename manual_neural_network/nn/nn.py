import numpy as np
from tqdm import tqdm


def softmax(a):
    """
    Softmax function that tries to avoid overflow by normalizing the input data

    :param a: array or list
    :return: softmax of a
    """
    a = np.asarray(a)
    exp_a = np.exp(a - a.max())
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def _forward_prop(model, X):
    """
    Calculate the probabilities of a feature through a forward pass

    :param model: dictionary containing the model parameters
    :param X: matrix of (n_samples x n_features)
    :return: vector of probabilities
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + np.repeat(b1, X.shape[0], axis=0)
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + np.repeat(b2, a1.shape[0], axis=0)

    return softmax(z2)


def calculate_loss(model: dict, X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the normalized negative log-loss

    :param model: dictionary containing the NN model
    :param X: data used for evaluating the model
    :param y: true labels corresponding to the rows of X
    :return: log-loss
    """
    probas = _forward_prop(model, X)

    # negative log loss
    log_probs = np.sum(- y * np.log(probas[:, y]))
    return float(log_probs / len(y))


def predict(model, X):
    """
    Predicts the labels of X given the model

    :param model: dictionary containing the NN model
    :param X: data used for evaluating the model
    :return: predicted labels of X
    """
    probas = _forward_prop(model, X)

    return np.argmax(probas)


def build_and_train(n_hidden: int, n_classes: int, X: np.ndarray, y: np.ndarray) -> dict:
    """

    :param n_hidden: number of nodes in the hidden layer
    :param n_classes: number of classes that should be predicted
    :param X: data used for evaluating the model
    :param y: true labels corresponding to the rows of X
    :return: dictionary of the model, containing the weights and biases of each layer
    """

    W1 = np.random.normal(loc=0, scale=1e-4, size=(X.shape[1], n_hidden)) / n_classes
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.normal(loc=0, scale=1e-4, size=(n_hidden, n_classes)) / n_hidden
    b2 = np.zeros((1, n_classes))

    epsilon = 0.00001
    reg_lambda = 0.001

    model = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    for epoch in tqdm(range(5000)):

        # forward pass
        z1 = X.dot(W1) + np.repeat(b1, X.shape[0], axis=0)
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + np.repeat(b2, a1.shape[0], axis=0)
        probas = softmax(z2)

        # backprop
        delta3 = probas.copy()
        delta3[:, y] -= y

        dW2 = a1.transpose().dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = (1 - np.power(a1, 2)) * (delta3.dot(W2.transpose()))

        dW1 = X.transpose().dot(delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # regularize
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2

        model = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }

        if epoch % 10 == 0:
            print("Loss after epoch %i: %f" % (epoch, calculate_loss(model, X, y)))

    return model

