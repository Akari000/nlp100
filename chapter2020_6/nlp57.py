'''57. 特徴量の重みの確認
52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，
重みの低い特徴量トップ10を確認せよ．
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from collections import Counter
import re


def tokenize(doc):
    doc = re.sub(r"[',.]", '', doc)  # 記号を削除
    tokens = doc.split(' ')
    tokens = [token.lower() for token in tokens]  # 小文字に統一
    return tokens


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

# vocabulary
train['tokens'] = train.title.apply(tokenize)
vocab = train['tokens'].tolist()
vocab = sum(vocab, [])  # flat list
counter = Counter(vocab)
vocab = [
    token
    for token, freq in counter.most_common()
    if 2 < freq < 300
]

# 学習
lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train, y_train)  # ロジスティック回帰モデルの重みを学習

labels = {'b': 0, 't': 1, 'e': 2, 'm': 3}

df = pd.DataFrame()
df['word'] = vocab

for key, value in labels.items():
    df['coef'] = lr.coef_[value]
    print(key)
    print('top', df.sort_values('coef', ascending=False).head(10).to_string(index=False), '\n')
    print('worst', df.sort_values('coef').head(10).to_string(index=False), '\n')

'''
b
top       word      coef
      bank  1.962625
       ecb  1.719458
       fed  1.689746
      euro  1.654264
   ukraine  1.638324
    stocks  1.609036
      debt  1.529753
    dollar  1.492231
     china  1.482172
 obamacare  1.439323

worst        word      coef
 activision -1.624413
      aereo -1.316694
       baby -1.201340
      ebola -1.193505
    gentiva -1.156369
      using -1.144158
        her -1.106829
      heart -1.105022
      virus -1.068223
      video -1.063307

t
top        word      coef
     google  2.920564
   facebook  2.845927
    climate  2.673567
  microsoft  2.436323
      apple  2.376684
    googles  2.014103
      tesla  1.998145
 heartbleed  1.983248
 activision  1.968263
         gm  1.862245

worst      word      coef
  percent -1.083380
     drug -1.080066
     body -1.035057
 american -0.971817
      fed -0.969407
     kids -0.960359
   cancer -0.955626
  thrones -0.953407
   stocks -0.940353
  healthy -0.912032

e
top        word      coef
      chris  1.720740
      movie  1.711239
 kardashian  1.600684
    thrones  1.577790
       paul  1.527678
    beyonce  1.455025
       film  1.452449
        kim  1.406111
    trailer  1.398830
      cyrus  1.391527

worst        word      coef
     google -1.495946
       data -1.295001
       risk -1.170615
         gm -1.166302
 scientists -1.131897
      china -1.079348
        oil -1.075257
      apple -1.066606
  microsoft -1.060468
      study -1.054368

m
top     word      coef
   ebola  2.929840
  cancer  2.746356
     fda  2.475477
    drug  2.406155
 doctors  2.255743
    mers  2.227592
 medical  2.167983
   cases  2.158971
   study  1.991106
 healthy  1.906352

worst      word      coef
 facebook -1.183311
      ceo -1.155707
       gm -1.148704
  climate -1.110640
    ocean -0.970629
    sales -0.956666
    apple -0.941436
     best -0.935091
     king -0.920177
     bank -0.886580
'''
