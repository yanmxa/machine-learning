import numpy as np

def accuracy_score(y, y_predict):
    assert len(y) == len(y_predict), 'the size of y must be equal to the size of y_predict'
    return np.sum(y == y_predict) / len(y)