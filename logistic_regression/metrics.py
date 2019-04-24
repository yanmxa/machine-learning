import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'the size of y_true must be equal to the size of y_predict'
    return np.sum(y_true == y_predict) / len(y_true)

'''
    线性回归衡量指标
        accuracy
        RMSE / MSE (Root Mean Squared Error) 
        MAE (Mean Absolute Error)
        R^2 = 1 - (MSE / var)
'''
def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict)**2) / len(y_true)

def root_mean_squared_error(y_true, y_predict):
    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)



'''
    分类问题衡量指标
        accuracy  准确率
        precision 精准率（针对某一类别而言）
        recall    召回率（针对某一类别而言）
        F1 score  精准率与召回率的调和平均
        precision recall curve 改变决策边界阈值，得到的不同精准率与召回率的组合
        ROC曲线    揭示TPR和FPR之间的关系  —— TPR与FPR正相关的理解：提高了召回率那么回水摸鱼的也就多了！
                   要想提高召回率的时候，掺杂的错误信息越少，
                   使用ROC曲线下面的面积 AUC [0,1]作为指标：TPR越大，FPR越小 => AUC越大
                   ROC和AUC对于有偏数据并不敏感，它主要应用于比较两个模型优劣
        
'''
def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))
def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))
def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))
def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))
def confusion_matrix(y_true, y_predict):
    # 真值\预测值(N,P)
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0
def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    try:
        return 2.*precision*recall/(precision+recall)
    except:
        return 0.


def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.

def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.


from sklearn.metrics import precision_score