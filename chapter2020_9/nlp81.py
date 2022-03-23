'''# 81. RNNによる予測
ID番号で表現された単語列x=(x1,x2,…,xT)がある．ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．
'''
import pandas as pd
import torch
from utils import preprocessor, tokens2ids
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import json


with open('token2id_dic.json', 'r') as f:
    token2id_dic = json.loads(f.read())
dw = 300
dh = 50
L = 4
columns = ('category', 'title')
vocab_size = len(token2id_dic)

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')


class RNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size, vocab_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb = torch.nn.Embedding(vocab_size, data_size)
        self.rnn = torch.nn.RNN(dw, dh, nonlinearity='relu')
        self.liner = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengs, hidden=None):   # x: (max_len)
        x = self.emb(x)                         # x: (max_length, dw)
        packed = pack_padded_sequence(
            x, lengs, batch_first=True, enforce_sorted=False)
        y, hidden = self.rnn(packed, hidden)    # y: (max_len, dh), hidden: (max_len, dh)
        # TODO yでなくhiddenを使えばpad_packed_sequenceをしなくて良い
        y, _ = pad_packed_sequence(y, batch_first=True)
        y = y[:, -1, :]
        y = self.liner(y)
        y = torch.softmax(y, dim=1)
        return y, hidden


train['tokens'] = train.title.apply(preprocessor)
X_train = tuple(train.tokens.apply(tokens2ids, token2id_dic=token2id_dic))
# X_train[0] = tensor([   8,    0, 2416, 1604, 2143,    5, 1605,    4,  745])

lengs = torch.tensor([len(x) for x in X_train])

inputs = pad_sequence(X_train, batch_first=True)
model = RNN(dw, dh, L, vocab_size)

outputs, hidden = model(inputs, lengs)

print(outputs.size())
print(hidden.size())

'''
torch.Size([10672, 4])
torch.Size([1, 10672, 50])
'''
