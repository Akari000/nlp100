# 38. ヒストグラム
# 単語の出現頻度のヒストグラム（横軸に出現頻度，縦軸に出現頻度をとる単語の種類数を棒グラフで表したもの）を描け．

import collections
import matplotlib.pyplot as plt
from nlp30 import get_morphs
morphs = []

with open('../data/neko.txt.mecab', 'r') as f:
    lines = f.readlines()
    morphs = get_morphs(lines)


surfaces = [morph['surface'] for morph in morphs]
c = collections.Counter(surfaces)

ranks = []
rank = 0
current = 0
for word in c.most_common():
    if current != word[1]:
        current = word[1]
        rank += 1
    ranks.append(rank)

plt.hist(ranks)
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
plt.savefig('../results/nlp39.png')
