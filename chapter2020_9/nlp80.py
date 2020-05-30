import pandas as pd
from collections import Counter
from utils import tokenize, normalize
import torch
import json


def token2id(token, token2id_dic):
    if token in token2id_dic:
        return token2id_dic[token]
    else:
        return 0


def tokens2ids(tokens, token2id_dic):
    tokens = [token2id(token, token2id_dic) for token in tokens]
    return torch.tensor(tokens, dtype=torch.long)


columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')


docs = [normalize(doc) for doc in train.title.values.tolist()]
tokens = [tokenize(doc) for doc in docs]
tokens = sum(tokens, [])  # flat list
counter = Counter(tokens)

token2id_dic = {}
vocab_size = len(counter)
for index, (token, freq) in enumerate(counter.most_common(), 1):
    if freq < 2:
        token2id_dic[token] = 0
    else:
        token2id_dic[token] = index

with open('token2id_dic.json', 'w') as f:
    json.dump(token2id_dic, f)


text = 'I am a cat'
ids = tokens2ids(tokenize(text), token2id_dic=token2id_dic)
print(ids)

'''
tensor([   0, 3353,   12, 3426])
'''
