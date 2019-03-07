import numpy as np
import matplotlib.pyplot as plt

def J(theta, X, y):
    try:
        return np.sum((y - X.dot(theta))**2) / len(X)
    except:
        return float('inf')             # 损失函数值太大产生溢出的话则返回一个最大值即可

def dJ(theta, X, y):
    grad = np.empty(len(theta))
    grad[0] = np.sum(X.dot(theta) - y)
    for i in range(1, len(theta)):
        grad[i] = (X.dot(theta) - y).dot(X[:,i])
    grad = grad * (2 / len(X))
    return grad

def gradient_descent(X, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    theta = initial_theta
    i_iters = 0
    while i_iters < n_iters:
        gradient = dJ(theta=theta, X=X, y=y)
        last_theta = theta
        theta = theta - eta * gradient
        if (abs(J(theta, X, y) - J(last_theta, X, y)) < epsilon):
            break
        i_iters += 1
    return theta

np.random.seed(666)
X = 2 * np.random.random(size=100)
y = X * 3. + 4. + np.random.normal(size=100)


X = np.hstack([ np.ones((len(X), 1)), X.reshape(-1, 1) ])
initial_theta = np.zeros(X.shape[1])
eta = 0.01
theta = gradient_descent(X=X, y=y, initial_theta = initial_theta, eta = eta)
print(theta)

