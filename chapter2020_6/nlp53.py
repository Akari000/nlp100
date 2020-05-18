'''53. 予測
52で学習したロジスティック回帰モデルを用い，
与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．
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
x_train = pd.read_csv('../data/NewsAggregatorDataset/train.feature.txt',
                      sep='\t', header=None)

# 学習
lr = LogisticRegression(class_weight='balanced')  # ロジスティック回帰モデルのインスタンスを作成
lr.fit(x_train, y_train)  # ロジスティック回帰モデルの重みを学習

# TODO 予測確率を出す


# 予測
def predict(x_):
    y_pred = lr.predict(x_)
    score = lr.predict_proba(x_)
    return (y_pred, score)


y_pred = predict(x_train.head(1))

print('タイトル', train.title.head(1).values)
print('予測ラベル', y_pred[0])
print('正解', y_train[0])
print('予測確率', y_pred[1])

'''
タイトル ["UPDATE 7-France's Iliad challenges Sprint for control of T-Mobile"]
予測ラベル [1]
正解 1
予測確率 [[0.18940928 0.71075679 0.02824979 0.07158414]]
'''
