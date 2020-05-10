'''51. 特徴量抽出Permalink
学習データ，検証データ，評価データから特徴量を抽出し，
それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ．
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．
'''

import pandas as pd

columns = ('id',
           'title',
           'url',
           'publisher',
           'category',
           'story',
           'hostname',
           'timestamp')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
valid = pd.read_csv('../data/NewsAggregatorDataset/valid.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')

train = train[['id', 'title', 'category', 'story']]
valid = valid[['id', 'title', 'category', 'story']]
test = test[['id', 'title', 'category', 'story']]

train.to_csv('../data/NewsAggregatorDataset/train.feature.txt',
             sep='\t', header=False, index=False)
valid.to_csv('../data/NewsAggregatorDataset/valid.feature.txt',
             sep='\t', header=False, index=False)
test.to_csv('../data/NewsAggregatorDataset/test.feature.txt',
            sep='\t', header=False, index=False)
