import numpy as np
import sklearn.datasets as ds
from sklearn.preprocessing import MinMaxScaler

from manual_neural_network.nn.nn import calculate_loss, build_and_train


def test_calc_loss_1():
    np.random.seed(42)

    X, y = [[1, 1]], [0]

    model = {
        'W1': [[1, 0, 1],
               [0, 1, 1]],
        'b1': [0, 0, 0],
        'W2': [[1, 0],
               [0, 1],
               [1, 1]],
        'b2': [0, 0]
    }

    np.testing.assert_approx_equal(calculate_loss(model, np.asarray(X), np.asarray(y)),
                                   0.6931471805599453)


def test_calc_loss():
    np.random.seed(42)

    X, y = ds.make_classification(n_samples=1000, n_features=7, n_informative=3,
                                  n_classes=3, random_state=42)
    model = {
        'W1': np.random.normal(loc=0, scale=0.1, size=(7, 10)),
        'b1': np.zeros((1, 10)),
        'W2': np.random.normal(loc=0, scale=0.1, size=(10, 3)),
        'b2': np.zeros((1, 3))
    }

    np.testing.assert_approx_equal(calculate_loss(model, X, y), 1100.3072278565282)


def test_build_and_train():
    np.random.seed(42)

    X, y = ds.make_classification(n_samples=1000, n_features=2, n_informative=2,
                                  n_classes=2, random_state=42, n_redundant=0)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = build_and_train(15, 2, X_scaled, y)
    print(model)
