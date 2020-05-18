'''51. 特徴量抽出Permalink
学習データ，検証データ，評価データから特徴量を抽出し，
それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ．
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．
'''

import pandas as pd
import re
from collections import Counter
from tqdm import tqdm
tqdm.pandas()


def tokenize(doc):
    doc = re.sub(r"[',.]", '', doc)  # 記号を削除
    tokens = doc.split(' ')
    tokens = [token.lower() for token in tokens]  # 小文字に統一
    return tokens


def preprocessor(tokens):
    tokens = [token for token in tokens if token in vocab]
    return tokens


def bag_of_words(doc):
    vector = [0]*len(vocab)
    for word in doc:
        if word in vocab:
            vector[vocab.index(word)] += 1
    return pd.Series(vector)


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

train['tokens'] = train.tokens.progress_apply(preprocessor)
x_train = train.tokens.progress_apply(bag_of_words)

test['tokens'] = test.title.apply(tokenize)
test['tokens'] = test.tokens.progress_apply(preprocessor)
x_test = test.tokens.progress_apply(bag_of_words)

valid['tokens'] = valid.title.apply(tokenize)
valid['tokens'] = valid.tokens.progress_apply(preprocessor)
x_valid = valid.tokens.progress_apply(bag_of_words)


x_train.to_csv('../data/NewsAggregatorDataset/train.feature.txt',
               sep='\t', header=False, index=False)
x_valid.to_csv('../data/NewsAggregatorDataset/valid.feature.txt',
               sep='\t', header=False, index=False)
x_test.to_csv('../data/NewsAggregatorDataset/test.feature.txt',
              sep='\t', header=False, index=False)
