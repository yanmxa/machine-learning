#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
    如何对文档进行分类 ？
'''

''' 1.对文档进行分词：英文分词ntlk, 中文jieba '''
import jieba
import os
LABEL_MAP = {'体育': 0, '女性': 1, '文学': 2, '校园': 3}
def load_data(dir_path):
    documents = []
    labels = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            label = root.split('/')[-1]
            labels.append(label)
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                content = f.read()
                world_list = list(jieba.cut(content))
                documents.append(' '.join(world_list))
    return documents, labels

train_contains, train_labels = load_data('./data/text_classification/train')
test_contains, test_labels = load_data('./data/text_classification/test')


''' 2.加载停用词表 '''
with open(r'./data/text_classification/stop/stopword.txt') as f:
    stop_words = [line.strip() for line in f.readlines()]


''' 3.计算单词权重 '''
from sklearn.feature_extraction.text import TfidfVectorizer
TF_IDF = TfidfVectorizer(stop_words=stop_words, max_df=0.5)  # max_df代表一个单词在50%的文档中都出现过，那么它携带的信息量就比较少，不作为分词统计
train_features = TF_IDF.fit_transform(train_contains)


''' 4.生成朴素贝叶斯分类器'''
# 多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)


''' 5.使用生成的分类器做预测'''
test_tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5, vocabulary=TF_IDF.vocabulary_)
test_features=test_tf.fit_transform(test_contains)
predicted_labels=clf.predict(test_features)

''' 6.计算准确率'''
from sklearn import metrics
print (metrics.accuracy_score(test_labels, predicted_labels))


