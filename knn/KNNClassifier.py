from collections import Counter
from math import sqrt
import numpy as np
from knn.metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k):
        assert k >= 1
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], 'the size of X_train must be equal to the size of y_train.'
        assert self.k <= X_train.shape[0], 'the size of X_train must be at least k.'
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None, 'must fit before predict.'
        assert X_predict.shape[1] == self._X_train.shape[1], 'the feature number of X_predict must be equal to X_train.'
        y_predict = [self._predict(x) for x in X_predict]
        return y_predict

    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1], 'the feature number of x must be equal to X_train.'
        distances = [sqrt(np.sum( (x_train - x)**2) ) for x_train in self._X_train]
        sorted_index = np.argsort(distances)
        topK_y = [self._y_train[i] for i in sorted_index[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return 'knn(k=%d)' % self.k


'''
    test the custom KNNClassifier with iris
'''
def iris():
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    cls = KNNClassifier(k=3)
    cls.fit(X_train, y_train)
    y_predict = cls.predict(X_test)
    print(np.sum(y_predict == y_test) / len(y_test))


'''
    test the custom KNNClassifier wit digits
'''
def digits():
    from sklearn import datasets
    from knn.KNNClassifier import KNNClassifier
    from knn.metrics import accuracy_score
    from knn.model_selection import train_test_split

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_rate=0.2)
    # digit_images = X[666].reshape(8,8)
    # plt.ion()
    # plt.imshow(digit_images, cmap=matplotlib.cm.binary)
    # plt.show()
    # plt.pause(1)

    custom_cls = KNNClassifier(3)
    custom_cls.fit(X_train, y_train)
    y_predict = custom_cls.predict(X_test)
    print(accuracy_score(y_test, y_predict))


'''
    sklearn 
'''
def sklearn():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  # 10 is seed

    '''
        以距离为权重， 1/distance 
        minkowski距离：
            曼哈顿距离           1 ----------- 1/n * sum(|x1 - x2|)               weights = uniform 不考虑权重
            欧拉距离             2 ----------- 1/n * sqrt.sum(|x1 - x2|^2)        weights = distance 考虑权重，默认是欧拉距离
            明可夫基斯距离 ：     n
        其他距离见 sklearn.neighbors.DistanceMetric
    '''
    # best_method = ''
    best_score = 0.0
    best_k = -1
    best_p = -1

    # for method in ['uniform', 'distance']:
    for p in range(1, 6):
        for k in range(1, 11):
            # cls = KNeighborsClassifier(n_neighbors=k , weights=method)
            cls = KNeighborsClassifier(n_neighbors=k, weights='distance', p=p)
            cls.fit(X_train, y_train)
            score = cls.score(X_test, y_test)
            if score > best_score:
                best_p = p
                best_k = k
                best_score = score

    print('best k = ', best_k)
    print('best p = ', best_p)
    print('best score = ', score)


'''
    sklearn Hyper-Parameter : grid search
'''
def sklearn_grid_search():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 10 is seed

    para_grid = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 11)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p': [i for i in range(1, 6)]
        }
    ]
    clf = KNeighborsClassifier()

    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(clf, para_grid, n_jobs=-1, verbose=2)  # n_jobs = 几个核进行运算 -1表示有几个用几个, verbose越大，输出信息越详细
    grid_search.fit(X_train, y_train)
    print(grid_search.best_estimator_)
    print(grid_search.best_params_)
    print(grid_search.best_index_)
    print(grid_search.best_score_)


'''
    feature scaling : 
        normalization - 若有outlier边界或者边界不明显则优势不明显
        standardization - 
'''
def feature_scaling():
    X = np.random.randint(0, 100, (50, 2))
    X = np.array(X, dtype=float)

    X[:, 0] = (X[:, 0] - np.min(X[:, 0])) / (np.max(X[:, 0]) - np.min(X[:, 0]))
    X[:, 1] = (X[:, 1] - np.min(X[:, 1])) / (np.max(X[:, 1]) - np.min(X[:, 1]))
    import matplotlib.pyplot as plt
    plt.ion()
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    plt.pause(2)

    X2 = np.random.randint(0, 10, (50, 2))
    X2 = np.array(X2, dtype=float)
    X2[:, 0] = (X2[:, 0] - np.mean(X2[:, 0])) / np.std(X2[:, 0])
    X2[:, 1] = (X2[:, 1] - np.mean(X2[:, 1])) / np.std(X2[:, 1])
    plt.scatter(X2[:, 0], X[:, 1])
    plt.show()
    plt.pause(10)

    print(np.mean(X2[:, 0]))
    print(np.std(X2[:, 0]))


'''
     sklearn scaler: 
'''
def sklearn_scaler():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train = standardScaler.transform(X_train)
    X_test = standardScaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)


if __name__ == '__main__':
    sklearn_scaler()