'''96. 学習過程の可視化
Tensorboardなどのツールを用い，ニューラル機械翻訳モデルが学習されていく過程を可視化せよ．可視化する項目としては，学習データにおける損失関数の値とBLEUスコア，開発データにおける損失関数の値とBLEUスコアなどを採用せよ．
'''
from torch.utils.tensorboard import SummaryWriter

# with open('../data/kftt-data-1.0/data/data') as f:
loss = [1, 2, 3]
bleu = [0.1, 0.2, 0.3]

writer = SummaryWriter(log_dir="./logs")

# xとyの値を記録していく
for i in range(len(loss)):
    writer.add_scalar("loss", loss[i], i)
    writer.add_scalar("bleu", bleu[i], i)

writer.close()

'''
1. $ tensorboard --logdir ./logs
2. localhost:6006 にアクセス