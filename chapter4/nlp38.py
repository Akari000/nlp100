# 38. ヒストグラム
# 単語の出現頻度のヒストグラム（横軸に出現頻度，縦軸に出現頻度をとる単語の種類数を棒グラフで表したもの）を描け．

import collections
import matplotlib.pyplot as plt
from nlp30 import get_morphs
morphs = []

# TODO グラフにラベルをつける


with open('../data/neko.txt.mecab', 'r') as f:
    lines = f.readlines()
    morphs = get_morphs(lines)

surfaces = [morph['surface'] for morph in morphs]
c = collections.Counter(surfaces)

count = [line[1] for line in c.most_common()]
plt.hist(count, range=(0, 50))
plt.savefig('../results/nlp38.png')
