'''54. 正解率の計測
52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression

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


#  訓練データの正解率
y_pred = lr.predict(x_train)
print('正解率', accuracy(y_pred, y_train))
'''
正解率 0.9845389805097451

'''

#  評価データの正解率
y_pred = lr.predict(x_test)
print('正解率', accuracy(y_pred, y_test))
'''
正解率 0.8958020989505248
'''
