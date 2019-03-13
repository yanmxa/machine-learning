import numpy as np
from knn.metrics import r2_score, accuracy_score

class LinearRegression():
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], 'the size of X_trian must be equal to the size of y_train.'
        X_b = np.hstack(np.ones((len(X_train), 1)), X_train)
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_grad(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], 'the size of X_trian must be equal to the size of y_train.'
        def J(theta, X, y):
            try:
                return np.sum((y - X.dot(theta)) ** 2) / len(X)
            except:
                return float('inf')

        def dJ(theta, X, y):
            grad = np.empty(len(theta))
            grad[0] = np.sum(X.dot(theta) - y)
            for i in range(1, len(theta)):
                grad[i] = (X.dot(theta) - y).dot(X[:, i])
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

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        assert X_train.shape[0] == y_train.shape[0], 'the size of X_trian must be equal to the size of y_train.'
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.0

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t1 + t)

            theta = initial_theta
            m = len(X_b)
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])

        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self


    def predict(self, X_predict):
        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return  X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return 'LinearRegression()'


'''
    sklearn中关于多元线性回归的使用
'''
def sklearn_linear_regression():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y < 50.0]
    y = y[y < 50.0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    print('>> score : ', lin_reg.score(X_test, y_test))

'''
    sklearn中使用KNN Regressor进行回归分析
'''
def KNN_regressioin():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y < 50.0]
    y = y[y < 50.0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    knn_reg = KNeighborsRegressor()
    knn_reg.fit(X_train, y_train)
    print('>> score : ', knn_reg.score(X_test, y_test))

'''
    sklearn中使用KNN Regressor进行回归分析, 使用网格搜索确定超参数
'''
def KNN_regressioin_param_grid():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y < 50.0]
    y = y[y < 50.0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    param_grid = [
        {
            'weights' : ['uniform'],
            'n_neighbors' : [ i for i in range(1, 11)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p' : [i for i in range(1, 6)]
        }
    ]
    knn_reg = KNeighborsRegressor()

    grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_.score(X_test, y_test))

'''
    测试梯度下降
'''
def linear_regression_grad():
    np.random.seed(666)
    x = 2 * np.random.random(size=100)
    y = x * 3. + 4. + np.random.normal(size=100)
    X = x.reshape(-1, 1)

    lrg = LinearRegression()
    lrg.fit_grad(X, y)
    print(lrg.coef_)
    print(lrg.interception_)

'''
    测试随机梯度下降
'''
def stochastic_grad():
    m = 10000
    x = np.random.normal(size=m)
    X = x.reshape(-1, 1)
    y = 4. * x + 3. + np.random.normal(0, 3, size=m)

    lin_reg = LinearRegression()
    lin_reg.fit_sgd(X, y, n_iters=2)

    print('-----------------------')
    from sklearn import datasets
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y < 50.0]
    y = y[y < 50.0]

    from knn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    from sklearn.preprocessing import StandardScaler
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train_standard = standardScaler.transform(X_train)
    X_test_standard = standardScaler.transform(X_test)

    lin_sgd = LinearRegression()
    lin_sgd.fit_sgd(X_train_standard, y_train, n_iters=50)

    print(lin_sgd.score(X_test_standard, y_test))

'''
    scikit-learn随机梯度下降
'''
def sklearn_stochastic_grad():

    from sklearn import datasets
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y < 50.0]
    y = y[y < 50.0]

    from knn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    from sklearn.preprocessing import StandardScaler
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train_standard = standardScaler.transform(X_train)
    X_test_standard = standardScaler.transform(X_test)

    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(n_iter=500)
    sgd_reg.fit(X_test_standard, y_test)
    print(sgd_reg.score(X_test_standard, y_test))

    lin_sgd = LinearRegression()
    lin_sgd.fit_sgd(X_train_standard, y_train, n_iters=500)
    print(lin_sgd.score(X_test_standard, y_test))

if __name__ == '__main__':
    sklearn_stochastic_grad()