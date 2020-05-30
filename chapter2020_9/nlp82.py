import pandas as pd
import torch
from utils import preprocessor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import detect_anomaly
from tqdm import tqdm
tqdm.pandas()


with open('token2id_dic.json', 'r') as f:
    token2id_dic = json.loads(f.read())
dw = 300
dh = 50
L = 4
batch_size = 1
columns = ('category', 'title')
vocab_size = len(token2id_dic)

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
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
        self.emb = nn.Embedding(vocab_size, data_size, padding_idx=0)
        self.rnn = nn.LSTM(dw, dh, batch_first=True)
        self.liner = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengs, hidden=None, cell=None):   # x: (max_len)
        x = self.emb(x)                         # x: (max_length, dw)
        packed = pack_padded_sequence(
            x, lengs, batch_first=True, enforce_sorted=False)
        y, (hidden, cell) = self.rnn(packed)    # y: (max_len, dh), hidden: (max_len, dh)
        y = self.liner(hidden.view(hidden.shape[1], -1))
        y = self.softmax(y)
        return y


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.lengs = torch.tensor([len(x) for x in data])
        self.data = pad_sequence(data, batch_first=True)
        self.labels = torch.tensor(labels).long()

        self.datanum = len(data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.labels[idx]
        lengs = self.lengs[idx]
        return out_data, out_label, lengs


train['tokens'] = train.title.apply(preprocessor)
test['tokens'] = test.title.apply(preprocessor)

X_train = train.tokens.apply(tokens2ids)
X_test = test.tokens.apply(tokens2ids)

label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
Y_train = train.category.map(label2int)
Y_test = test.category.map(label2int)


# TODO 評価データでも正解率を求める．

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(dw, dh, L, vocab_size)
criterion = nn.CrossEntropyLoss()  # クロスエントロピー損失関数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 確率的勾配降下法

trainset = Mydatasets(X_train, Y_train)
loader = DataLoader(trainset, batch_size=batch_size)

model = model.to(device)
ds_size = len(trainset)
nan_inputs = 0
for epoch in range(10):
    n_correct = 0
    total_loss = 0
    for i, (inputs, labels, lengs) in enumerate(tqdm(loader)):
        inputs = inputs[:, :max(lengs)]
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengs = lengs.to(device)

        # with detect_anomaly():
        outputs = model(inputs, lengs)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        outputs = np.argmax(outputs.data.numpy(), axis=1)
        labels = labels.data.numpy()
        print(len(labels), len(outputs))
        for output, label in zip(outputs, labels):
            if output == label:
                n_correct += 1

        print('correct', n_correct)
    print('ds', ds_size)
    print('epoch: %d loss: %f accuracy: %f' % (
      epoch, loss, n_correct/ds_size))


print('Finished Training')
'''
epoch: 0 loss: 1.366026 accuracy: 0.611694
epoch: 1 loss: 2.227322 accuracy: 0.717860
epoch: 2 loss: 2.329761 accuracy: 0.755903
epoch: 3 loss: 1.523480 accuracy: 0.781016
epoch: 4 loss: 2.170162 accuracy: 0.791042
epoch: 5 loss: 0.075167 accuracy: 0.799475
'''
