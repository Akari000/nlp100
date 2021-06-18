'''87. 確率的勾配降下法によるCNNの学習
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，
問題86で構築したモデルを学習せよ．訓練データ上の損失と正解率，
評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．
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


class CNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size, vocab_size):
        super(CNN, self).__init__()
        self.emb = torch.nn.Embedding(vocab_size, data_size)
        self.conv = torch.nn.Conv1d(data_size, hidden_size, 3, padding=1)  # in_channels, out_channels, kernel_sizes
        self.pool = torch.nn.MaxPool1d(120)
        self.liner_px = nn.Linear(data_size*3, hidden_size)
        self.liner_yc = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):                       # x: (max_len)
        x = self.emb(x)                         # x: (max_length, dw)
        x = x.view(-1, x.shape[2], x.shape[1])  # x: (dw, max_length)
        x = self.conv(x)                        # 畳み込み x: (dh, max_len)
        p = self.act(x)
        c = self.pool(p)                        # c: (dh, 1)
        c = c.view(c.shape[0], c.shape[1])      # c: (1, dh)
        y = self.liner_yc(c)                    # c: (1, L)
        y = torch.softmax(y, dim=1)
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
batch_size = 8
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
model = CNN(dw, dh, L, vocab_size)
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
[train]	loss: 1.347948 accuracy: 0.405547
[test]	loss: 1.316919 accuracy: 0.437031
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:27<00:00, 49.27it/s]
epoch:  1
[train]	loss: 1.326331 accuracy: 0.417166
[test]	loss: 1.291175 accuracy: 0.439280
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:28<00:00, 46.84it/s]
epoch:  2
[train]	loss: 1.316955 accuracy: 0.422695
[test]	loss: 1.280001 accuracy: 0.438531
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:26<00:00, 50.93it/s]
epoch:  3
[train]	loss: 1.311620 accuracy: 0.429442
[test]	loss: 1.274323 accuracy: 0.429535
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:28<00:00, 47.54it/s]
epoch:  4
[train]	loss: 1.307364 accuracy: 0.438156
[test]	loss: 1.271154 accuracy: 0.420540
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:25<00:00, 52.93it/s]
epoch:  5
[train]	loss: 1.303161 accuracy: 0.450993
[test]	loss: 1.269326 accuracy: 0.414543
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:32<00:00, 41.57it/s]
epoch:  6
[train]	loss: 1.298589 accuracy: 0.460551
[test]	loss: 1.268325 accuracy: 0.418291
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:27<00:00, 49.06it/s]
epoch:  7
[train]	loss: 1.293372 accuracy: 0.469828
[test]	loss: 1.267901 accuracy: 0.418291
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:25<00:00, 52.16it/s]
epoch:  8
[train]	loss: 1.287272 accuracy: 0.480978
[test]	loss: 1.267904 accuracy: 0.416042
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:28<00:00, 47.51it/s]
epoch:  9
[train]	loss: 1.280119 accuracy: 0.489505
[test]	loss: 1.268249 accuracy: 0.415292
Finished Training
'''