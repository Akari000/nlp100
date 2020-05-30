import pandas as pd
import torch
from utils import preprocessor
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


def token2id(token):
    if token in token2id_dic:
        return token2id_dic[token]
    else:
        return 0


def tokens2ids(tokens):
    tokens = [token2id(token) for token in tokens]
    return torch.tensor(tokens, dtype=torch.long)


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
        y, _ = pad_packed_sequence(y, batch_first=True)
        y = y[:, -1, :]
        y = self.liner(y)
        y = torch.softmax(y, dim=1)
        return y, hidden


train['tokens'] = train.title.apply(preprocessor)
X_train = train.tokens.apply(tokens2ids)
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
