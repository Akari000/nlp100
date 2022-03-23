'''85. 双方向RNN・多層化
順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習せよ．
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

class BidirectionalRNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size, vocab_size):
        super(BidirectionalRNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, data_size, padding_idx=0)
        self.rnn1 = torch.nn.LSTM(data_size, hidden_size, bidirectional=True)
        self.liner = nn.Linear(2*hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengs, hidden=None, cell=None):   # x: (max_len)
        x = self.emb(x)                         # x: (max_length, dw)
        packed = pack_padded_sequence(
            x, lengs, batch_first=True, enforce_sorted=False)
        y, (hidden, cell) = self.rnn1(packed)    # y: (max_len, dh), hidden: (max_len, dh)
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


with open('token2id_dic.json', 'r') as f:
    token2id_dic = json.loads(f.read())

dw = 300
dh = 50
L = 4
batch_size =128
columns = ('category', 'title')
vocab_size = len(token2id_dic)

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')


train['tokens'] = train.title.apply(preprocessor)
test['tokens'] = test.title.apply(preprocessor)

X_train = tuple(train.tokens.apply(tokens2ids, token2id_dic=token2id_dic))
X_test = tuple(test.tokens.apply(tokens2ids, token2id_dic=token2id_dic))

label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
Y_train = train.category.map(label2int)
Y_test = test.category.map(label2int)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BidirectionalRNN(dw, dh, L, vocab_size)
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
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 71.06it/s]
epoch:  0
[train]	loss: 1.320634 accuracy: 0.420446
[test]	loss: 1.267285 accuracy: 0.403298
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 73.02it/s]
epoch:  1
[train]	loss: 1.319298 accuracy: 0.481166
[test]	loss: 1.271431 accuracy: 0.415292
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 71.97it/s]
epoch:  2
[train]	loss: 1.321238 accuracy: 0.545915
[test]	loss: 1.279712 accuracy: 0.416792
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 73.59it/s]
epoch:  3
[train]	loss: 1.283335 accuracy: 0.614130
[test]	loss: 1.290095 accuracy: 0.410795
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 72.70it/s]
epoch:  4
[train]	loss: 1.172834 accuracy: 0.675881
[test]	loss: 1.299543 accuracy: 0.404048
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 73.12it/s]
epoch:  5
[train]	loss: 1.070396 accuracy: 0.721233
[test]	loss: 1.306725 accuracy: 0.402549
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 73.23it/s]
epoch:  6
[train]	loss: 1.041079 accuracy: 0.750469
[test]	loss: 1.308352 accuracy: 0.396552
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 73.82it/s]
epoch:  7
[train]	loss: 1.001567 accuracy: 0.765367
[test]	loss: 1.312892 accuracy: 0.393553
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:18<00:00, 71.16it/s]
epoch:  8
[train]	loss: 1.014474 accuracy: 0.776237
[test]	loss: 1.315099 accuracy: 0.394303
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:17<00:00, 74.27it/s]
epoch:  9
[train]	loss: 0.998510 accuracy: 0.782421
[test]	loss: 1.316557 accuracy: 0.397301
Finished Training
'''