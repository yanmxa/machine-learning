import numpy as np
from knn.custom import KNNClassifier
from sklearn import datasets

'''
    test the custom KNNClassifier with iris
'''
def iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    cls =  KNNClassifier(k=3)
    cls.fit(X_train, y_train)
    y_predict = cls.predict(X_test)
    print(np.sum(y_predict == y_test) / len(y_test))

'''
    test the custom KNNClassifier wit digits
'''
def digits():
    from sklearn import datasets
    from knn.custom import KNNClassifier
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
    from sklearn.metrics import accuracy_score

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) # 10 is seed

    cls = KNeighborsClassifier(n_neighbors=3)
    cls.fit(X_train, y_train)
    y_predict = cls.predict(X_test)
    print(accuracy_score(y_predict, y_test))


if __name__ == '__main__':
    sklearn()



