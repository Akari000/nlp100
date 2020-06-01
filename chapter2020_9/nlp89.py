'''# 89. 事前学習済み言語モデルからの転移学習
事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
'''
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
tqdm.pandas()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False,)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds.detach().numpy(), axis=1)
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def trainer(model, optimizer, loader, test_loader, ds_size, device, max_iter=10):
    for epoch in range(max_iter):
        n_correct = 0
        total_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss, logits = model(inputs, labels=labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # count loss and n_correct
            total_loss += loss.data
            logits = np.argmax(logits.data.numpy(), axis=1)
            labels = labels.data.numpy()
            for logit, label in zip(logits, labels):
                if logit == label:
                    n_correct += 1

            print('loss: %f\t n_correct: %d' % (loss.data, n_correct))

        print('epoch: ', epoch)
        print('[train]\tloss: %f accuracy: %f' % (loss, n_correct/ds_size))

        test_loss, test_acc = evaluate(model, test_loader)
        print('[test]\tloss: %f accuracy: %f' % (test_loss, test_acc))

    print('Finished Training')


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        # self.data = pad_sequence(data, batch_first=True)
        # max_len = len(self.data[0])
        # self.mask = torch.tensor([[1]*len(x)+[0]*(max_len-len(x)) for x in data])
        self.labels = torch.tensor(labels).long()
        self.datanum = len(data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.labels[idx]
        # mask = self.mask[idx]
        # print(out_data)
        return out_data, out_label


def normalize(doc):
    doc = re.sub(r"[',.]", '', doc)   # 記号を削除
    doc = re.sub(r" {2,}", ' ', doc)  # 2回以上続くスペースを削除
    doc = re.sub(r" *?$", '', doc)    # 行頭と行末のスペースを削除
    doc = re.sub(r"^ *?", '', doc)
    doc = doc.lower()                 # 小文字に統一
    return doc


def preprocessor(doc, max_len):
    doc = normalize(doc)
    tokens = tokenizer.encode_plus(
        doc,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_tensors='pt')
    tokens = tokens['input_ids']
    return tokens.squeeze(0)


def accuracy(pred, label):
    pred = np.argmax(pred.data.numpy(), axis=1)  # 行ごとに最大値のインデックスを取得する．
    label = label.data.numpy()
    return (pred == label).mean()


def evaluate(model, loader, criterion):
    for inputs, labels, lengs in loader:
        outputs = model(inputs, lengs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
    return loss.data, acc


dw = 300
dh = 50
L = 4
batch_size = 32
columns = ('category', 'title')

train = pd.read_csv('../data/NewsAggregatorDataset/train.txt',
                    names=columns, sep='\t')
test = pd.read_csv('../data/NewsAggregatorDataset/test.txt',
                   names=columns, sep='\t')


# get max seequence length
# train['tokens'] = train.title.apply(normalize)
# train['tokens'] = train.tokens.apply(tokenizer.tokenize)

# train['len'] = train.tokens.apply(len)
# max_len = train[train.len < 512].sort_values(
#     'len', ascending=False).len.tolist()[0]
max_len = 31
train['tokens'] = train.title.apply(preprocessor, max_len=max_len)
test['tokens'] = test.title.apply(preprocessor, max_len=max_len)


X_train = train['tokens'].values.tolist()
X_test = train['tokens'].values.tolist()


label2int = {'b': 0, 't': 1, 'e': 2, 'm': 3}
Y_train = train.category.map(label2int)
Y_test = test.category.map(label2int)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 確率的勾配降下法

trainset = Mydatasets(X_train, Y_train)
testset = Mydatasets(X_test, Y_test)
loader = DataLoader(trainset, batch_size=batch_size)
test_loader = DataLoader(testset, batch_size=testset.__len__())

model = model.to(device)
ds_size = trainset.__len__()

trainer(model, optimizer, loader, test_loader, ds_size, device)
