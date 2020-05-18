'''52. 学習
51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression

columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
valid = pd.read_csv('../data/NewsAggregatorDataset/valid.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')


label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = train['category'].map(label2int)  # クラスを定義
y_valid = valid['category'].map(label2int)
y_test = test['category'].map(label2int)
del train, valid, test

x_train = pd.read_csv('../data/NewsAggregatorDataset/train.feature.txt',
                      sep='\t', header=None)
x_valid = pd.read_csv('../data/NewsAggregatorDataset/valid.feature.txt',
                      sep='\t', header=None)
x_test = pd.read_csv('../data/NewsAggregatorDataset/test.feature.txt',
                     sep='\t', header=None)


lr = LogisticRegression(class_weight='balanced')  # ロジスティック回帰モデルのインスタンスを作成
lr.fit(x_train, y_train)  # ロジスティック回帰モデルの重みを学習
