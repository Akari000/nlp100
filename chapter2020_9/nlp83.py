'''# 83. ミニバッチ化・GPU上での学習
問題82のコードを改変し，B事例ごとに損失・勾配を計算して学習を行えるようにせよ（Bの値は適当に選べ）．
また，GPU上で学習を実行せよ．
'''
import pandas as pd
from utils import preprocessor, tokens2ids
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
import json
from tqdm import tqdm
tqdm.pandas()


with open('token2id_dic.json', 'r') as f:
    token2id_dic = json.loads(f.read())

dw = 300
dh = 50
L = 4
batch_size = 1024
columns = ('category', 'title')
vocab_size = len(token2id_dic)

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')


def accuracy(pred, label):
    pred = np.argmax(pred.data.numpy(), axis=1)  # 行ごとに最大値のインデックスを取得する．
    label = label.data.numpy()
    return (pred == label).mean()


def evaluate(model, loader):
    for inputs, labels, lengs in loader:
        outputs = model(inputs, lengs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
    return loss.data, acc


def trainer(model, criterion, optimizer, loader, test_loader, ds_size, device, max_iter=10):
    for epoch in range(10):
        n_correct = 0
        total_loss = 0
        for i, (inputs, labels, lengs) in enumerate(tqdm(loader)):
            inputs = inputs[:, :max(lengs)]
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengs = lengs.to(device)

            outputs = model(inputs, lengs)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            outputs = np.argmax(outputs.data.numpy(), axis=1)
            labels = labels.data.numpy()
            for output, label in zip(outputs, labels):
                if output == label:
                    n_correct += 1

        print('epoch: ', epoch)
        print('[train]\tloss: %f accuracy: %f' % (loss, n_correct/ds_size))

        test_loss, test_acc = evaluate(model, test_loader)
        print('[test]\tloss: %f accuracy: %f' % (test_loss, test_acc))

    print('Finished Training')


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

X_train = train.tokens.apply(tokens2ids, token2id_dic=token2id_dic)
X_test = test.tokens.apply(tokens2ids, token2id_dic=token2id_dic)

label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
Y_train = train.category.map(label2int)
Y_test = test.category.map(label2int)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(dw, dh, L, vocab_size)
criterion = nn.CrossEntropyLoss()  # クロスエントロピー損失関数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 確率的勾配降下法

trainset = Mydatasets(X_train, Y_train)
testset = Mydatasets(X_test, Y_test)
loader = DataLoader(trainset, batch_size=batch_size)
test_loader = DataLoader(testset, batch_size=testset.__len__())

model = model.to(device)
ds_size = trainset.__len__()

trainer(model, criterion, optimizer, loader, test_loader, ds_size, device, 10)

'''
epoch:  0
[train]	loss: 1.381180 accuracy: 0.286451
[test]	loss: 1.385540 accuracy: 0.294603
epoch:  1
[train]	loss: 1.380485 accuracy: 0.293010
[test]	loss: 1.384784 accuracy: 0.299850
epoch:  2
[train]	loss: 1.379793 accuracy: 0.300319
[test]	loss: 1.384027 accuracy: 0.307346
epoch:  3
[train]	loss: 1.379105 accuracy: 0.307534
[test]	loss: 1.383276 accuracy: 0.313343
epoch:  4
[train]	loss: 1.378418 accuracy: 0.315592
[test]	loss: 1.382528 accuracy: 0.317841
epoch:  5
[train]	loss: 1.377735 accuracy: 0.322058
[test]	loss: 1.381781 accuracy: 0.326087
epoch:  6
[train]	loss: 1.377054 accuracy: 0.328898
[test]	loss: 1.381039 accuracy: 0.329085
epoch:  7
[train]	loss: 1.376375 accuracy: 0.336207
[test]	loss: 1.380298 accuracy: 0.337331
epoch:  8
[train]	loss: 1.375700 accuracy: 0.340611
[test]	loss: 1.379560 accuracy: 0.339580
epoch:  9
[train]	loss: 1.375027 accuracy: 0.345577
[test]	loss: 1.378823 accuracy: 0.341079
Finished Training
'''
