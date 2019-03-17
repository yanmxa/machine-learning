import numpy as np
import matplotlib.pyplot as plt

def dJ(theta):
    return 2 * (theta - 2.5)

def J(theta):
    try:
        return (theta - 2.5) ** 2 -1
    except:
        return float('inf')

def gradient_descent(initial_theta, eta, n_iters=1e4, epsilon=1e-8):
    theta = initial_theta
    theta_history = [initial_theta]
    i_iters = 0
    while i_iters < n_iters:
        gradient = dJ(theta=theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        if (abs(J(theta) - J(last_theta)) < epsilon):
            break
        i_iters += 1
    return theta_history

def dJ_debug(theta, X_b, y, epsilon=0.01):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        res[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y)) / (2 * epsilon)
    return res

def plot_theta_history(plot_x, theta_history):
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
    plt.show()

def test():
    eta = 0.01
    theta_history = gradient_descent(0, eta, n_iters=10)
    plot_x = np.linspace(-1, 6, 141)
    plot_theta_history(plot_x, theta_history)


if __name__ == '__main__':
    test()
