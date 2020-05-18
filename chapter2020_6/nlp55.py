'''55. 混同行列の作成
52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，
学習データおよび評価データ上で作成せよ．
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')

label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = train['category'].map(label2int)  # クラスを定義
y_test = test['category'].map(label2int)

x_train = pd.read_csv('../data/NewsAggregatorDataset/train.feature.txt',
                      sep='\t', header=None)
x_test = pd.read_csv('../data/NewsAggregatorDataset/test.feature.txt',
                     sep='\t', header=None)

# 学習
lr = LogisticRegression(class_weight='balanced')  # ロジスティック回帰モデルのインスタンスを作成
lr.fit(x_train, y_train)  # ロジスティック回帰モデルの重みを学習


# 予測
def predict_category(x_):
    y_pred = lr.predict(x_test)
    return y_pred


def accuracy(predict, y):
    return (predict == y).mean()


def confusion_matrix(y_true, y_pred):
    size = len(set(y_true))
    result = np.array([0]*(size*size)).reshape((size, size))  # 配列の初期化
    for t, p in zip(y_true, y_pred):
        result[t][p] += 1
    return result


#  訓練データの混同行列
y_pred = lr.predict(x_train)
con_matrix = confusion_matrix(y_train, y_pred)
print(con_matrix)
'''
[[4410   73   20   16]
 [   5 1203    2    0]
 [  13   11 4182    8]
 [   0    0    0  729]]
'''

#  評価データの混同行列
y_pred = lr.predict(x_test)
con_matrix = confusion_matrix(y_test, y_pred)
print(con_matrix)
'''
[[498  36  10   9]
 [ 21 124   4   4]
 [ 14  14 505   5]
 [  7   8   7  68]]
'''
