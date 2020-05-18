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

# TODO tokenize をpreprocessorにわける
# TODO preprocessorをreduce_vocabに変える


def preprocessor(doc):
    doc = re.sub(r"[',.]", '', doc)     # 記号を削除
    doc = doc.lower()             # 小文字に統一
    return doc


def tokenize(doc):
    tokens = doc.split(' ')
    return tokens


def reduce_vocab(tokens):
    tokens = [token for token in tokens if token in vocab]
    return tokens


def bag_of_words(doc):
    vector = [0]*len(vocab)
    for word in doc:
        if word in vocab:
            vector[vocab.index(word)] += 1
    return pd.Series(vector)


columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
valid = pd.read_csv('../data/NewsAggregatorDataset/valid.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')

# preprocess
train['tokens'] = train.title.progress_apply(preprocessor)
test['tokens'] = test.title.progress_apply(preprocessor)
valid['tokens'] = valid.title.progress_apply(preprocessor)

# tokenize
train['tokens'] = train.tokens.apply(tokenize)
test['tokens'] = test.tokens.apply(tokenize)
valid['tokens'] = valid.tokens.apply(tokenize)

# reduce vocabulary
vocab = train['tokens'].tolist()
vocab = sum(vocab, [])  # flat list
counter = Counter(vocab)
vocab = [
    token
    for token, freq in counter.most_common()
    if 2 < freq < 300
]

train['tokens'] = train.tokens.apply(reduce_vocab)
test['tokens'] = test.tokens.apply(reduce_vocab)
valid['tokens'] = valid.tokens.apply(reduce_vocab)

x_train = train.tokens.progress_apply(bag_of_words)
x_test = test.tokens.progress_apply(bag_of_words)
x_valid = valid.tokens.progress_apply(bag_of_words)

x_train.to_csv('../data/NewsAggregatorDataset/train.feature.txt',
               sep='\t', header=False, index=False)
x_valid.to_csv('../data/NewsAggregatorDataset/valid.feature.txt',
               sep='\t', header=False, index=False)
x_test.to_csv('../data/NewsAggregatorDataset/test.feature.txt',
              sep='\t', header=False, index=False)
