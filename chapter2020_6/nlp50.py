'''50. データの入手・整形Permalink
News Aggregator Data Setをダウンロードし、以下の要領で学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

1. ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
2. 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
3. 抽出された事例をランダムに並び替える．
4. 抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，
それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ
（このファイルは後に問題70で再利用する）．
'''
# TODO 要素数をカウントする．
# TODO カテゴリ名と記事見出しのタブ区切りで保存する

import pandas as pd
import numpy as np
columns = ('id',
           'title',
           'url',
           'publisher',
           'category',
           'story',
           'hostname',
           'timestamp')
publisher = ('Reuters', 'Huffington Post',
             'Businessweek', 'Contactmusic.com', 'Daily Mail')

posts = pd.read_csv('../data/NewsAggregatorDataset/newsCorpora.csv',
                    names=columns, sep='\t')
posts = posts[posts.publisher.isin(publisher)]
posts = posts[['category', 'title']]
posts = posts.sample(frac=1)  # ランダムサンプリング
size = len(posts)
train, valid, test = np.split(posts, [int(.8*size), int(.9*size)])  # 8:1:1に分割

train.to_csv('../data/NewsAggregatorDataset/train.txt', sep='\t', header=False, index=False)
valid.to_csv('../data/NewsAggregatorDataset/valid.txt', sep='\t', header=False, index=False)
test.to_csv('../data/NewsAggregatorDataset/test.txt', sep='\t', header=False, index=False)
