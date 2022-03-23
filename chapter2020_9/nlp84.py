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

googlenews_vectors = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)
# weights = googlenews.syn0

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
    def __init__(self, vocab_size, data_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, data_size)
        # self.emb.weight = torch.nn.Parameter(torch.from_numpy(weights))  # TODO embeddingの重みをgooglenews.weightで初期化する．tokenを照しあわせる
        self.hidden_size = hidden_size
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
batch_size = 128
columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')


train['tokens'] = train.title.apply(preprocessor)
test['tokens'] = test.title.apply(preprocessor)

X_train = tuple(train.tokens.apply(tokens2ids, token2id_dic=token2id_dic))
X_test = tuple(test.tokens.apply(tokens2ids, token2id_dic=token2id_dic))

pad_sequence(X_train, batch_first=True)
label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
Y_train = train.category.map(label2int)
Y_test = test.category.map(label2int)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(len(token2id_dic), dw, dh, L)

for token, i in token2id_dic.items():
    if token in googlenews_vectors:
        model.emb.weight.data[i] = torch.tensor(googlenews_vectors[token], dtype=torch.float32)

# 学習されるパラメータにする
model.emb.weight = torch.nn.Parameter(model.emb.weight)
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
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 102.30it/s]
epoch:  0
[train]	loss: 1.326823 accuracy: 0.405922
[test]	loss: 1.299863 accuracy: 0.407796
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 102.33it/s]
epoch:  1
[train]	loss: 1.298807 accuracy: 0.425319
[test]	loss: 1.265611 accuracy: 0.410045
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 102.16it/s]
epoch:  2
[train]	loss: 1.279971 accuracy: 0.461957
[test]	loss: 1.234034 accuracy: 0.503748
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 101.37it/s]
epoch:  3
[train]	loss: 1.245675 accuracy: 0.637463
[test]	loss: 1.159216 accuracy: 0.705397
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 100.89it/s]
epoch:  4
[train]	loss: 1.073214 accuracy: 0.733977
[test]	loss: 1.006331 accuracy: 0.758621
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 98.35it/s]
epoch:  5
[train]	loss: 1.005972 accuracy: 0.757871
[test]	loss: 0.984906 accuracy: 0.766117
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 101.04it/s]
epoch:  6
[train]	loss: 0.995498 accuracy: 0.766679
[test]	loss: 0.977657 accuracy: 0.767616
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 101.28it/s]
epoch:  7
[train]	loss: 0.993898 accuracy: 0.773238
[test]	loss: 0.973279 accuracy: 0.775112
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 102.57it/s]
epoch:  8
[train]	loss: 0.994754 accuracy: 0.777361
[test]	loss: 0.970067 accuracy: 0.777361
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1334/1334 [00:13<00:00, 100.22it/s]
epoch:  9
[train]	loss: 0.990764 accuracy: 0.781203
[test]	loss: 0.967038 accuracy: 0.778861
Finished Training
'''