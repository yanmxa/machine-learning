'''
 sklearn 提供了撒各个朴素贝叶斯分类算法
        1) 高斯朴素贝叶斯(GaussianNB)：特征变量是连续变量
        2) 伯努利朴素贝叶斯(BernouliNB)：特征变量是布尔值，符合0/1分布
        3）多项式朴素贝叶斯(MultinomialNB)：特征变量是离散变量，符合多项式分布

    文本分类：TF-IDF值 Term Frequency - Inverse Document Frequency
             TF = 单次出现的次数 / 该文档的总单词数
             IDF = log(文档总数/(该单词出现的文档数)+1)         sklearn中用的是ln
             TF-IDF = TF * IDF

'''
from sklearn.feature_extraction.text import TfidfVectorizer
TF_IDF_vec = TfidfVectorizer()
documents = [
    'this is the bayes document',
    'this is the second second document',
    'and the third one',
    'is this the document'
]
TF_IDF_matrix = TF_IDF_vec.fit_transform(documents)
print('不重复的词:', TF_IDF_vec.get_feature_names())
print('每个单词的 ID:', TF_IDF_vec.vocabulary_)
print('每个单词的 tfidf 值:')
for l in TF_IDF_matrix.toarray():
    print(l)



