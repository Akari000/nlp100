'''56. 適合率，再現率，F1スコアの計測
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
カテゴリごとに適合率，再現率，F1スコアを求め，カテゴリごとの性能をマイクロ平均（micro-average）
とマクロ平均（macro-average）で統合せよ．
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train, y_train)  # ロジスティック回帰モデルの重みを学習


#  評価データの混同行列
y_pred = lr.predict(x_test)
labels = [0, 1, 2, 3]

print('presotion', precision_score(y_test, y_pred, average=None, labels=labels))
print('recall', recall_score(y_test, y_pred, average=None, labels=labels))
print('f1_score', f1_score(y_test, y_pred, average=None, labels=labels))

print('マイクロ平均')  # 値は全て一致する．
print('presition', precision_score(y_test, y_pred, average='micro', labels=labels))
print('recall', recall_score(y_test, y_pred, average='micro', labels=labels))
print('f1_score', f1_score(y_test, y_pred, average='micro', labels=labels))

print('マクロ平均')
print('presition', precision_score(y_test, y_pred, average='macro', labels=labels))
print('recall', recall_score(y_test, y_pred, average='macro', labels=labels))
print('f1_score', f1_score(y_test, y_pred, average='macro', labels=labels))

'''
presotion	 [0.92558984 0.74545455 0.94392523 0.75903614]
recall	 [0.89788732 0.80921053 0.95283019 0.75      ]
f1_score	 [0.91152815 0.77602524 0.94835681 0.75449102]
マイクロ平均
presition 0.9002998500749625
recall 0.9002998500749625
f1_score 0.9002998500749625
マクロ平均
presition 0.8435014400845838
recall 0.8524820097346741
f1_score 0.8476003030507292
'''
