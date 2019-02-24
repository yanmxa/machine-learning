import numpy as np

def train_test_split(X, y, test_rate = 0.2, seed=None):
    assert X.shape[0] == y.shape[0], 'the size of X must be equal to the size of y.'
    assert 0.0 <= test_rate <= 1.0, 'test_rate must be valid.'
    if seed:
        np.random.seed(seed)

    shuffle_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * 0.2)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test