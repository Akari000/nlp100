'''# 84. 単語ベクトルの導入
事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）での学習済み単語ベクトル）で単語埋め込みemb(x)を初期化し，学習せよ．
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
from gensim.models import KeyedVectors
from tqdm import tqdm
tqdm.pandas()


googlenews = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)
weights = googlenews.wv.syn0


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
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.emb.weight = torch.nn.Parameter(weights)  # TODO embeddingの重みをgooglenews.weightで初期化する
        self.hidden_size = hiden_size
        self.rnn = nn.LSTM(data_size, hidden_size, batch_first=True)
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


with open('token2id_dic.json', 'r') as f:
    token2id_dic = json.loads(f.read())

dw = 300
dh = 50
L = 4
batch_size = 8
columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')


train['tokens'] = train.title.apply(preprocessor)
test['tokens'] = test.title.apply(preprocessor)

X_train = tuple(train.tokens.apply(tokens2ids, token2id_dic=token2id_dic))
X_test = tuple(test.tokens.apply(tokens2ids, token2id_dic=token2id_dic))

# X_train = torch.tensor(X_train, dtype=torch.float32)

pad_sequence(X_train, batch_first=True)
label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
Y_train = train.category.map(label2int)
Y_test = test.category.map(label2int)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(dw, dh, L)
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
[train]	loss: 1.306197 accuracy: 0.418103
[test]	loss: 1.296853 accuracy: 0.437781
epoch:  1
[train]	loss: 1.278086 accuracy: 0.422695
[test]	loss: 1.258812 accuracy: 0.456522
epoch:  2
[train]	loss: 1.253694 accuracy: 0.561282
[test]	loss: 1.212802 accuracy: 0.682909
epoch:  3
[train]	loss: 1.220054 accuracy: 0.735757
[test]	loss: 1.014457 accuracy: 0.766867
epoch:  4
[train]	loss: 1.233165 accuracy: 0.765555
[test]	loss: 0.978168 accuracy: 0.774363
epoch:  5
[train]	loss: 1.236411 accuracy: 0.772114
[test]	loss: 0.968411 accuracy: 0.780360
epoch:  6
[train]	loss: 1.238182 accuracy: 0.776143
[test]	loss: 0.964659 accuracy: 0.781859
epoch:  7
[train]	loss: 1.238690 accuracy: 0.778579
[test]	loss: 0.959650 accuracy: 0.784108
epoch:  8
[train]	loss: 1.240388 accuracy: 0.781765
[test]	loss: 0.958462 accuracy: 0.784108
epoch:  9
[train]	loss: 1.241063 accuracy: 0.783265
[test]	loss: 0.957121 accuracy: 0.785607
Finished Training
'''