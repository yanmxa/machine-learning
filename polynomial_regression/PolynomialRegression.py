import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class PolyRegression:
    def __int__(self, degree):
        assert degree <= 0, 'degree must be valid !'
        self.poly = Pipeline([
            ("Poly", PolynomialFeatures(degree)),
            ("std", StandardScaler()),
            ("linear", LinearRegression())
        ])

    def fit(self, X, y):
        self.poly.fit(X, y)
        return self

    def predict(self, X):
        return self.poly.predict(X)
