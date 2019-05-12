# -*- coding: utf-8 -*-
'''
    ensemble : Boosting / Bagging
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def regression_boston():
    # 加载数据
    data = load_boston()
    # 分割数据
    train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.25, random_state=33)

    # 使用 AdaBoost 回归模型
    regressor = AdaBoostRegressor()
    regressor.fit(train_x, train_y)
    pred_y = regressor.predict(test_x)
    mse = mean_squared_error(test_y, pred_y)
    # print(" 房价预测结果 ", pred_y)
    print("AdaBoost均方误差 = ", round(mse, 2))

    # 使用决策树回归模型
    dec_regressor = DecisionTreeRegressor()
    dec_regressor.fit(train_x, train_y)
    pred_y = dec_regressor.predict(test_x)
    mse = mean_squared_error(test_y, pred_y)
    print(" 决策树均方误差 = ", round(mse, 2))

    # 使用 KNN 回归模型
    knn_regressor = KNeighborsRegressor()
    knn_regressor.fit(train_x, train_y)
    pred_y = knn_regressor.predict(test_x)
    mse = mean_squared_error(test_y, pred_y)
    print("KNN 均方误差 = ", round(mse, 2))

def classify_digits():
    # 加载数据
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt

    digits = load_digits()
    data = digits.data
    print(data.shape)
    print(digits.images[0])
    print(digits.target[0])
    # 将第一幅图像显示出来
    plt.gray()
    plt.imshow(digits.images[0])
    plt.show()

    # 分割数据，将25%的数据作为测试集，其余作为训练集
    train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

    # 采用Z-Score规范化
    ss = preprocessing.StandardScaler()
    train_ss_x = ss.fit_transform(train_x)
    test_ss_x = ss.transform(test_x)

    # 创建KNN分类器
    knn = KNeighborsClassifier()
    knn.fit(train_ss_x, train_y)
    predict_y = knn.predict(test_ss_x)
    print("KNN准确率: %.4lf" % accuracy_score(predict_y, test_y))

    # 创建SVM分类器
    svm = SVC()
    svm.fit(train_ss_x, train_y)
    predict_y = svm.predict(test_ss_x)
    print('SVM准确率: %0.4lf' % accuracy_score(predict_y, test_y))

    # 创建CART决策树分类器
    dtc = DecisionTreeClassifier()
    dtc.fit(train_ss_x, train_y)
    predict_y = dtc.predict(test_ss_x)
    print("CART决策树准确率1: %.4lf" % accuracy_score(predict_y, test_y))

    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    train_mm_x = mm.fit_transform(train_x)
    test_mm_x = mm.transform(test_x)

    # 创建Naive Bayes分类器
    mnb = MultinomialNB()
    mnb.fit(train_mm_x, train_y)
    predict_y = mnb.predict(test_mm_x)
    print("多项式朴素贝叶斯准确率: %.4lf" % accuracy_score(predict_y, test_y))


def comparator():
    import numpy as np
    from sklearn import datasets
    from sklearn.metrics import zero_one_loss
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    import matplotlib.pyplot as plt
    # 设置 AdaBoost 迭代次数
    n_estimators = 200
    # 使用
    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
    # 从 12000 个数据中取前 2000 行作为测试集，其余作为训练集
    train_x, train_y = X[2000:], y[2000:]
    test_x, test_y = X[:2000], y[:2000]
    # 弱分类器
    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    dt_stump.fit(train_x, train_y)
    dt_stump_err = 1.0 - dt_stump.score(test_x, test_y)
    # 决策树分类器
    dt = DecisionTreeClassifier()
    dt.fit(train_x, train_y)
    dt_err = 1.0 - dt.score(test_x, test_y)
    # AdaBoost 分类器
    ada = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=n_estimators)
    ada.fit(train_x, train_y)
    # 三个分类器的错误率可视化
    fig = plt.figure()
    # 设置 plt 正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax = fig.add_subplot(111)
    print([dt_stump_err]*2)
    print(dt_err)
    ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-', label=u'Decision Stump')
    ax.plot([1, n_estimators], [dt_err] * 2, 'k--', label=u'Decision Tree')
    ada_err = np.zeros((n_estimators,))
    # 遍历每次迭代的结果 i 为迭代次数, pred_y 为预测结果
    for i, pred_y in enumerate(ada.staged_predict(test_x)):
        # 统计错误率
        ada_err[i] = zero_one_loss(pred_y, test_y)
    # 绘制每次迭代的 AdaBoost 错误率
    ax.plot(np.arange(n_estimators) + 1, ada_err, label='AdaBoost', color='orange')
    ax.set_xlabel('times')
    ax.set_ylabel('error rate')
    leg = ax.legend(loc='upper right', fancybox=True)
    plt.show()


comparator()