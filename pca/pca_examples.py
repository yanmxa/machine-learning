import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

def mnist_knn():
    mnist = fetch_mldata('MNIST original', data_home='./datasets')
    X, y = mnist['data'], mnist['target']
    X_train = np.array(X[:60000], dtype=float)
    y_train = np.array(y[:60000], dtype=float)
    X_test = np.array(X[60000:], dtype=float)
    y_test = np.array(X[60000:], dtype=float)
    # knn_clf = KNeighborsClassifier()
    # start = time.time()
    # knn_clf.fit(X_train, y_train)
    # print('knn fitting without pca, cost time:', time.time() - start, 's')
    pca = PCA(0.9)
    pca.fit(X_train)
    X_train_reduction = pca.transform(X_train)
    X_test_reduction = pca.transform(X_test)
    knn_clf = KNeighborsClassifier()

    print('knn starting ...')
    start = time.time()
    knn_clf.fit(X_train_reduction, y_train)
    print('knn fitting with pca, cost time:', time.time() - start, 's')
    score = knn_clf.score(X_test_reduction, y_test)
    print('knn prediction accuracy', score)

def denoise():
    def plot_digits(data):
        fig, axes = plt.subplots(ncols=10,
                                 nrows=10,
                                 figsize=(10, 10),
                                 subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw = dict(hspace=0.05, wspace=0.05))
        for i, ax in enumerate(axes.flat):
            ax.imshow(data[i].reshape(8,8),
                      cmap='binary',
                      interpolation='nearest',
                      clim=(0, 16))
        # plt.show()

    from sklearn import datasets
    digits =datasets.load_digits()
    X = digits.data
    y = digits.target
    noisy_digits = X + np.random.normal(0, 4, size=X.shape)
    example_digits = noisy_digits[y==0,:][:10]
    for num in range(1, 10):
        X_num = noisy_digits[y==num, :][:10]
        example_digits = np.vstack([example_digits, X_num])

    plt.ion()
    plot_digits(example_digits)
    plt.pause(5)

    pca = PCA(0.5)
    pca.fit(noisy_digits)
    example_digits_reduction = pca.transform(example_digits)
    filtered_digits = pca.inverse_transform(example_digits_reduction)

    plot_digits(filtered_digits)
    plt.pause(10)

def face_feature():
    from sklearn.datasets import fetch_lfw_people
    faces = fetch_lfw_people(data_home='./datasets')



if __name__ == '__main__':
    face_feature()



