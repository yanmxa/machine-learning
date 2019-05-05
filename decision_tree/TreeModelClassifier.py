from collections import Counter
from math import log
import numpy as np

class TreeModelClassifier:
    def __init__(self, criterion='entropy', max_depth=20):
        def entropy(y):
            counter = Counter(y)
            res = 0.0
            for num in counter.values():
                p = num / len(y)
                res += -p * log(p)
            return res
        self.criterion = eval(criterion)
        self.rootNode = None
        self.maxDepth = max_depth
        self.countDepth = -1

    def fit(self, X, y):
        assert len(X) > 0 and len(X) == len(y), 'The X or y must be valid.'
        X = np.array(X)
        y = np.array(y)
        self.rootNode = self._constructTree(X, y)
        return self

    def predict(self, X):
        assert self.rootNode is not None, 'must fit before predict.'
        assert len(X) > 0, 'len(X) should > 0.'
        y_predict = [self._predict(x) for x in X]
        return np.array(y_predict)

    class Node:
        def __init__(self, col=-1, value=None, leaf=None, trueBranch=None, falseBranch=None):
            self.col = col
            self.value = value
            self.leaf = leaf
            self.trueBranch = trueBranch
            self.falseBranch = falseBranch

    def _divideData(self, X_column, value):
        splitFn = None
        if isinstance(value, int) or isinstance(value, float):
            splitFn = lambda x : x >= value
        else:
            splitFn = lambda x : x == value
        trueIndex = [i for i in range(0, len(X_column)) if splitFn(X_column[i])]
        falseIndex = [i for i in range(0, len(X_column)) if not splitFn(X_column[i])]
        return (trueIndex, falseIndex)

    def _constructTree(self, X_, y_):
        assert len(X_) == len(y_), 'len(X), len(y) must be equal.'
        if len(X_) == 0: return TreeModelClassifier.Node()
        self.countDepth += 1
        currentImpurity = self.criterion(y_)
        bestGain, bestSplits, bestIndexes = 0, None, None
        for col in range(0, len(X_[0])):
            colCounter = Counter(X_[:,col])
            for value in colCounter:
                (trueIndex, falseIndex) = self._divideData(X_[:,col], value)
                trueProb = float(len(trueIndex)) / len(X_)
                nextImpurity = trueProb * self.criterion(y_[trueIndex]) \
                               + (1-trueProb) * self.criterion(y_[falseIndex])
                gain = currentImpurity - nextImpurity
                if gain > bestGain and len(trueIndex) > 0 and len(falseIndex) > 0:
                    bestGain, bestSplits, bestIndexes = gain, (col, value), (trueIndex, falseIndex)
        if bestGain > 0 and len(bestIndexes[0]) > 0 and len(bestIndexes[1]) > 0 and self.countDepth < self.maxDepth:
            trueBranch = self._constructTree(X_[bestIndexes[0]].copy(), y_[bestIndexes[0]].copy())
            falseBranch = self._constructTree(X_[bestIndexes[1]].copy(), y_[bestIndexes[1]].copy())
            return TreeModelClassifier.Node(col=bestSplits[0], value=bestSplits[1], trueBranch=trueBranch, falseBranch=falseBranch)
        else:
            return TreeModelClassifier.Node(leaf=Counter(y_))

    def _predict(self, x):
        tree = self.rootNode
        leafCounter = self._classify(x, tree)
        return leafCounter.most_common(1)[0][0]

    def _classify(self, x, tree):
        if tree.leaf != None: return tree.leaf
        else:
            value = x[tree.col]
            branch = None
            if isinstance(value, int) or isinstance(value, float):
                if value >= tree.value: branch = tree.trueBranch
                else: branch = tree.falseBranch
            else:
                if value == tree.value: branch == tree.trueBranch
                else: branch = tree.falseBranch
            return self._classify(x, branch)


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)

import matplotlib.pyplot as plt
from sklearn import datasets

def IrisTest():
    iris = datasets.load_iris()
    X = iris.data[:,2:]
    y = iris.target
    from sklearn.tree import DecisionTreeClassifier
    dt_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")
    dt_clf.fit(X, y)
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plot_decision_boundary(dt_clf, axis=[0.5, 7.5, 0, 3])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.scatter(X[y==2,0], X[y==2,1])
    plt.title("scikit-learn")
    clt = TreeModelClassifier(max_depth=2)
    clt.fit(X, y)
    plt.subplot(122)
    plot_decision_boundary(clt, axis=[0.5, 7.5, 0, 3])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.scatter(X[y==2,0], X[y==2,1])
    plt.title("TreeModelClassifier")
    plt.show()

def MoonTest():
    X, y = datasets.make_moons(noise=0.25, random_state=666)
    from sklearn.tree import DecisionTreeClassifier
    dt_clf = DecisionTreeClassifier(criterion="entropy")
    dt_clf.fit(X, y)
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plot_decision_boundary(dt_clf, axis=[-1.5,2.5,-1.0,1.5])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.scatter(X[y==2,0], X[y==2,1])
    plt.title("scikit-learn")
    clt = TreeModelClassifier()
    clt.fit(X, y)
    plt.subplot(122)
    plot_decision_boundary(clt, axis=[-1.5,2.5,-1.0,1.5])
    plt.scatter(X[y==0,0], X[y==0,1])
    plt.scatter(X[y==1,0], X[y==1,1])
    plt.scatter(X[y==2,0], X[y==2,1])
    plt.title("TreeModelClassifier")
    plt.show()

MoonTest()