'''59. ハイパーパラメータの探索
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np


# TODO 他のハイパーパラメータも変えてみる


columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')
valid = pd.read_csv('../data/NewsAggregatorDataset/valid.txt',
                    names=columns, sep='\t')

x_train = pd.read_csv('../data/NewsAggregatorDataset/train.feature.txt',
                      sep='\t', header=False, index=False)
x_test = pd.read_csv('../data/NewsAggregatorDataset/test.feature.txt',
                     sep='\t', header=False, index=False)
x_valid = pd.read_csv('../data/NewsAggregatorDataset/valid.feature.txt',
                      sep='\t', header=False, index=False)

y_train = train['category'].map({'b': 0, 't': 1, 'e': 2, 'm': 3})  # クラスを定義
y_test = test['category'].map({'b': 0, 't': 1, 'e': 2, 'm': 3})
y_valid = valid['category'].map({'b': 0, 't': 1, 'e': 2, 'm': 3})

# 学習
lr = LogisticRegression(class_weight='balanced')  # ロジスティック回帰モデルのインスタンスを作成
lr.fit(x_train, y_train)  # ロジスティック回帰モデルの重みを学習


# 予測
def accuracy(predict, y):
    return (predict == y).mean()


# アルゴリズムの比較
# ロジスティック回帰
y_pred = lr.predict(x_train)
print('正解率', accuracy(y_pred, y_train))
'''
0.8958020989505248
'''

# SVM（サポートベクターマシン）
svm = SVC()
svm.fit(x_train, y_train)
score = svm.score(x_valid, y_valid)
print('正解率: {}' .format(score))
'''
正解率: 0.881559220389805
'''

# K-NearestNeighbor（K近傍法）
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
score = knn.score(x_valid, y_valid)
print('正解率: {}' .format(score))
'''
正解率: 0.5524737631184408
'''

# ランダムフォレスト
random_forest = DecisionTreeClassifier()
random_forest.fit(x_train, y_train)
score = random_forest.score(x_valid, y_valid)
print('正解率: {}' .format(score))
'''
正解率: 0.8118440779610195
'''

# パラメータの選択
x = []
y = []
params = np.logspace(-2, 3, 10, base=10)
for param in params:
    # 学習
    lr = LogisticRegression(class_weight='balanced', C=param)
    lr.fit(x_train, y_train)
    #  予測
    y_pred = lr.predict(x_valid)

    x.append(param)
    y.append(accuracy(y_pred, y_valid))

index = np.argmax(y)
c = x[index]
lr = LogisticRegression(class_weight='balanced', C=c)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('正規化パラメータc: %f, 検証データの正解率: %f' % (c, y[index]))
print('正解率: %f' % (accuracy(y_pred, y_test)))
'''
正規化パラメータc: 0.464159, 検証データの正解率: 0.905547
正解率: 0.898801
'''
