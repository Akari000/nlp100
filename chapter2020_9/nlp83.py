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
batch_size = 128
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

# パラメータを調整する
def trainer(model, criterion, optimizer, loader, test_loader, ds_size, device, max_iter=10):
    for epoch in range(20):
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
            x, lengs.cpu(), batch_first=True, enforce_sorted=False)
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

X_train = tuple(train.tokens.apply(tokens2ids, token2id_dic=token2id_dic))
X_test = tuple(test.tokens.apply(tokens2ids, token2id_dic=token2id_dic))

label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
Y_train = train.category.map(label2int)
Y_test = test.category.map(label2int)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(dw, dh, L, vocab_size)
criterion = nn.CrossEntropyLoss()  # クロスエントロピー損失関数
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 確率的勾配降下法

trainset = Mydatasets(X_train, Y_train)
testset = Mydatasets(X_test, Y_test)
loader = DataLoader(trainset, batch_size=batch_size)
test_loader = DataLoader(testset, batch_size=testset.__len__())

model = model.to(device)
ds_size = trainset.__len__()

trainer(model, criterion, optimizer, loader, test_loader, ds_size, device, 10)

'''
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:03<00:00, 27.99it/s]
epoch:  0
[train]	loss: 1.340198 accuracy: 0.360851
[test]	loss: 1.342829 accuracy: 0.402549
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.18it/s]
epoch:  1
[train]	loss: 1.308863 accuracy: 0.440499
[test]	loss: 1.309498 accuracy: 0.432534
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.12it/s]
epoch:  2
[train]	loss: 1.288382 accuracy: 0.457834
[test]	loss: 1.287992 accuracy: 0.445277
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.31it/s]
epoch:  3
[train]	loss: 1.270628 accuracy: 0.473951
[test]	loss: 1.270902 accuracy: 0.477511
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.00it/s]
epoch:  4
[train]	loss: 1.251525 accuracy: 0.505903
[test]	loss: 1.254446 accuracy: 0.519490
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.02it/s]
epoch:  5
[train]	loss: 1.229449 accuracy: 0.534951
[test]	loss: 1.237132 accuracy: 0.553973
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 28.72it/s]
epoch:  6
[train]	loss: 1.204475 accuracy: 0.560251
[test]	loss: 1.218670 accuracy: 0.580210
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 28.65it/s]
epoch:  7
[train]	loss: 1.177660 accuracy: 0.583208
[test]	loss: 1.199370 accuracy: 0.597451
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.03it/s]
epoch:  8
[train]	loss: 1.150552 accuracy: 0.602043
[test]	loss: 1.179888 accuracy: 0.611694
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 28.97it/s]
epoch:  9
[train]	loss: 1.124441 accuracy: 0.621908
[test]	loss: 1.160845 accuracy: 0.626687
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.44it/s]
epoch:  10
[train]	loss: 1.100010 accuracy: 0.640367
[test]	loss: 1.142638 accuracy: 0.642429
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.47it/s]
epoch:  11
[train]	loss: 1.077358 accuracy: 0.655454
[test]	loss: 1.125356 accuracy: 0.654423
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.49it/s]
epoch:  12
[train]	loss: 1.056060 accuracy: 0.669228
[test]	loss: 1.108791 accuracy: 0.667916
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 28.39it/s]
epoch:  13
[train]	loss: 1.035558 accuracy: 0.686282
[test]	loss: 1.092645 accuracy: 0.679910
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.75it/s]
epoch:  14
[train]	loss: 1.015878 accuracy: 0.701462
[test]	loss: 1.076754 accuracy: 0.695652
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.69it/s]
epoch:  15
[train]	loss: 0.998363 accuracy: 0.713549
[test]	loss: 1.061443 accuracy: 0.709145
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.91it/s]
epoch:  16
[train]	loss: 0.984377 accuracy: 0.724232
[test]	loss: 1.047296 accuracy: 0.721889
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.79it/s]
epoch:  17
[train]	loss: 0.973463 accuracy: 0.735851
[test]	loss: 1.034401 accuracy: 0.730885
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.36it/s]
epoch:  18
[train]	loss: 0.964800 accuracy: 0.744753
[test]	loss: 1.022182 accuracy: 0.734633
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 84/84 [00:02<00:00, 29.18it/s]
epoch:  19
[train]	loss: 0.957116 accuracy: 0.756747
[test]	loss: 1.011754 accuracy: 0.740630
Finished Training
'''
