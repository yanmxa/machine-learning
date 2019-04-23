import numpy as np

class PCA:
    def __init__(self, n_components):
        assert n_components >= 1, 'n_components must be valid !'
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        assert self.n_components <= X.shape[1], 'n_components must not be greater than the feature number of X'

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):

            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1
            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i,:] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1,1) * w

        return self

    def transform(self, X):
        ''' 将给定的X，映射到各个主成分分量中 '''
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        ''' 将给定的X，映射回原来的特征空间 '''
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    def __repr__(self):
        return 'PCA(n_components=%d)' % self.n_components

def PCA_test():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0., 10., size=100)

    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.components_)

    pca = PCA(1)
    pca.fit(X)
    X_reduction = pca.transform(X)
    print(X_reduction.shape)

    X_restore = pca.inverse_transform(X_reduction)
    print(X_restore.shape)

    import matplotlib.pyplot as plt
    plt.scatter(X[:,0], X[:,1], color='b', alpha=0.3)
    plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
    plt.show()

def scikit_learn_pca_test():
    from sklearn.decomposition import PCA

    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100., size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0., 10., size=100)

    pca = PCA(n_components=1)
    pca.fit(X)
    X_reduction = pca.transform(X)
    X_restore = pca.inverse_transform(X_reduction)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], color='b', alpha=0.3)
    plt.scatter(X_restore[:, 0], X_restore[:, 1], color='r', alpha=0.5)
    plt.show()

def scikit_learn_pca_digits():
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import time

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    from sklearn.neighbors import KNeighborsClassifier
    start = time.time()
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    end = time.time()
    print('digits classifier without pca, cost time :', (end-start)*1000, 'ms')
    print('digits accuracy :', knn_clf.score(X_test, y_test))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_train_reduction = pca.transform(X_train)
    X_test_reduction = pca.transform(X_test)

    start = time.time()
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_reduction, y_train)
    end = time.time()
    print('digits classifier with pca, cost time :', (end - start) * 1000, 'ms')
    print('digits accuracy :', knn_clf.score(X_test_reduction, y_test))

    '''对于pca中主成分数量的选取（n_components取值），
    sklearn中为我们提供了一个explain_variance_ration_的取值，
    表示每一个主成分占原数据方差的多少, 或者说保留了原数据多少的信息。'''
    pca = PCA(n_components=X_train.shape[1])
    pca.fit(X_train)
    plt.plot([i for i in range(X_train.shape[1])],
             [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
    plt.show()

    ''' sklearn 为我们提供了一个接口，只需要提供保留原来信息的比例，对于n_components的数值自行选择'''
    pca = PCA(0.95) # 保留原数据95%以上的方差信息
    pca.fit(X_train)
    print(pca.n_components_)
    X_train_reduction = pca.transform(X_train)
    X_test_reduction = pca.transform(X_test)

    start = time.time()
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_reduction, y_train)
    end = time.time()
    print('digits classifier with pca, cost time :', (end - start) * 1000, 'ms')
    print('digits accuracy :', knn_clf.score(X_test_reduction, y_test))

def decomposition_2D_vision():
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.decomposition import PCA

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    pca = PCA(n_components=2)
    pca.fit(X)
    X_reduction = pca.transform(X)
    for i in range(10):
        plt.scatter(X_reduction[y==i,0], X_reduction[y==i, 1], alpha=0.5)
    plt.show()

if __name__ == '__main__':
    decomposition_2D_vision()