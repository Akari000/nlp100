'''86. 畳み込みニューラルネットワーク (CNN)
ID番号で表現された単語列x=(x1,x2,…,xT)がある．ただし，Tは単語列の長さ，
xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
畳み込みニューラルネットワーク（CNN: Convolutional Neural Network）を用い，
単語列xからカテゴリyを予測するモデルを実装せよ
'''
import pandas as pd
from utils import preprocessor, tokens2ids
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
from tqdm import tqdm
tqdm.pandas()


def accuracy(pred, label):
    pred = np.argmax(pred.data.numpy(), axis=1)  # 行ごとに最大値のインデックスを取得する．
    label = label.data.numpy()
    return (pred == label).mean()


def evaluate(model, loader):
    for inputs, labels in loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
    return loss.data, acc


def trainer(model, criterion, optimizer, loader, test_loader, ds_size, device, max_iter=10):
    for epoch in range(10):
        n_correct = 0
        total_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs[:, :max(lengs)]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
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
        filter_size = 3
        self.emb = torch.nn.Embedding(vocab_size, data_size)
        # self.conv = torch.nn.Conv1d(data_size, hidden_size, 3, padding=1)  # in_channels, out_channels, kernel_sizes
        self.conv = torch.nn.Conv2d(1, hidden_size, (filter_size, data_size), stride=1, padding=(1,0))
        self.pool = torch.nn.MaxPool1d(120)
        self.liner_px = nn.Linear(data_size*3, hidden_size)
        self.liner_yc = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):                       # x: (max_len)
        x = self.emb(x).unsqueeze(1)                         # x: (max_length, dw)
        # x = x.view(-1, x.shape[2], x.shape[1])  # x: (dw, max_length)
        x = self.conv(x)                       # 畳み込み x: (dh, max_len)
        x = x.unsqueeze(-1)
        p = self.act(x)
        c = self.pool(p)                        # c: (dh, 1)
        c = c.view(c.shape[0], c.shape[1])      # c: (1, dh)
        y = self.liner_yc(c)                    # c: (1, L)
        y = torch.softmax(y, dim=1)
        return y


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = pad_sequence(data, batch_first=True)
        self.labels = torch.tensor(labels).long()

        self.datanum = len(data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.labels[idx]
        return out_data, out_label


with open('token2id_dic.json', 'r') as f:
    token2id_dic = json.loads(f.read())

dw = 300
dh = 50
L = 4
batch_size = 8
vocab_size = len(token2id_dic)
columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')

train['tokens'] = train.title.apply(preprocessor)


X_train = tuple(train.tokens.apply(tokens2ids, token2id_dic=token2id_dic))


model = CNN(dw, dh, L, vocab_size)
inputs = pad_sequence(X_train, batch_first=True)
outputs = model(inputs[:1])

print('output.size', outputs.size())
print(outputs)

'''
output.size torch.Size([1, 4])
tensor([[0.2001, 0.3345, 0.3549, 0.1105]], grad_fn=<SoftmaxBackward>)
'''