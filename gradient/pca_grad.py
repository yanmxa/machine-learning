import numpy as np
import matplotlib.pyplot as plt

def demean(X):
    return X - np.mean(X, axis=0)

def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def df_debug(w, X, epsilon=0.0001):
    res = np.empty(len(w))
    for i in range(len(w)):
        w1 = w.copy()
        w1[i] += epsilon
        w2 = w.copy()
        w2[i] -= epsilon
        res[i] = (f(w1, X) - f(w2, X)) / (2 * epsilon)
    return res

def direction(w):
    return w / np.linalg.norm(w)   # w的意义只是表示方向，故转换为单位矩阵

def first_component(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    cur_iters = 0
    w = direction(initial_w)
    while cur_iters < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break
        cur_iters += 1
    return w

def first_n_component(n, X, eta=0.01, n_iters=1e4, epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(df=df, X=X_pca, initial_w=initial_w, eta=eta, n_iters=n_iters, epsilon=epsilon)
        res.append(w)
        X_pca = X_pca - X_pca.dot(w).reshape(-1,1) * w
    return res

def data_test():
    X = np.empty((100,2))
    X[:,0] = np.random.uniform(0., 100., size=100)
    X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0., 10., size=100)

    plt.ion()
    plt.scatter(X[:,0], X[:,1])
    plt.pause(5)

    X_demean = demean(X)
    plt.scatter(X_demean[:,0], X_demean[:,1])
    plt.pause(5)

    print(np.mean(X_demean[:,0]), np.mean(X_demean[:,1]))

def first_component_test():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0., 10., size=100)

    initial_w = np.random.random(X.shape[1])
    eta = 0.001

    X_demean = demean(X)
    w = first_component(df, X_demean, initial_w, eta)
    # w_debug = first_component(df_debug, X_demean, initial_w, eta)
    # plt.plot(w_debug, w)
    # plt.show()
    plt.scatter(X_demean[:,0], X_demean[:,1])
    plt.plot([0, w[0]*30], [0, w[1]*30], color='r')
    plt.show()


def n_component_test():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0., 10., size=100)

    initial_w = np.random.random(X.shape[1])
    eta = 0.01
    # X_demean = demean(X)

    w = first_component(df, X, initial_w, eta)
    X2 = X - X.dot(w).reshape(-1,1) * w
    # X2 = np.empty(X.shape)
    # for i in range(len(X)):
    #     X2[i] = X[i] - X[i].dot(w) * w
    plt.scatter(X2[:,0], X2[:,1])
    plt.show()

def first_n_component_test():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0., 10., size=100)

    w = first_n_component(2, X)
    print(w)
    print(w[0].dot(w[1]))

if __name__ == '__main__':
    first_n_component_test()
