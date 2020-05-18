'''58. 正則化パラメータの変更Permalink
ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，
学習時の過学習（overfitting）の度合いを制御できる．
異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．
実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
'''

from matplotlib import pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')

x_train = pd.read_csv('../data/NewsAggregatorDataset/train.feature.txt',
                      sep='\t', header=False, index=False)
x_test = pd.read_csv('../data/NewsAggregatorDataset/test.feature.txt',
                     sep='\t', header=False, index=False)

y_train = train['category'].map({'b': 0, 't': 1, 'e': 2, 'm': 3})  # クラスを定義
y_test = test['category'].map({'b': 0, 't': 1, 'e': 2, 'm': 3})


def accuracy(predict, y):
    return (predict == y).mean()

# TODO 学習データ，検証データでもプロットする
x = []
y = []
params = np.logspace(-2, 3, 10, base=10)
for param in params:
    # 学習
    lr = LogisticRegression(class_weight='balanced', C=param)
    lr.fit(x_train, y_train)
    #  予測
    y_pred = lr.predict(x_test)

    x.append(param)
    y.append(accuracy(y_pred, y_test))

plt.plot(x, y)
plt.xlabel('正規化パラメータ')
plt.ylabel('正解率')
plt.title('正則化パラメータと正解率')
plt.savefig('../results/nlp2020_58test.png')
