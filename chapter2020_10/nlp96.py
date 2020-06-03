'''96. 学習過程の可視化
Tensorboardなどのツールを用い，ニューラル機械翻訳モデルが学習されていく過程を可視化せよ．可視化する項目としては，学習データにおける損失関数の値とBLEUスコア，開発データにおける損失関数の値とBLEUスコアなどを採用せよ．
'''
from torch.utils.tensorboard import SummaryWriter
import re

# 学習のコマンドでtensorboardのオプションをつけられるらしい．
with open('./demo-model-log') as f:
    text = f.read()

steps = re.findall(r'Step (\d+)/', text)
accuracy = re.findall(r'acc: {1,3}(.*?);', text)
loss = re.findall(r'xent: (.*?);', text)

steps = [int(step) for step in steps]
accuracy = [float(acc) for acc in accuracy]
loss = [float(l) for l in loss]


writer = SummaryWriter(log_dir="./logs")

for i, step in enumerate(steps):
    writer.add_scalar("loss", loss[i], step)
    writer.add_scalar("accuracy", accuracy[i], step)

writer.close()

'''
1. $ tensorboard --logdir ./logs
2. localhost:6006 にアクセス
'''
