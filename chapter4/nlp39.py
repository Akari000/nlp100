# 38. ヒストグラム
# 単語の出現頻度のヒストグラム（横軸に出現頻度，縦軸に出現頻度をとる単語の種類数を棒グラフで表したもの）を描け．

import pickle
import collections
import matplotlib.pyplot as plt

morpheme_list = []
with open('../data/neko.txt.mecab', 'rb') as f:
    morpheme_list = pickle.load(f)

words = [node['surface'] for node in morpheme_list]
c = collections.Counter(words)

rank_list = []
rank = 0
current = 0
for word in c.most_common():
    if current != word[1]:
        current = word[1]
        rank += 1
    rank_list.append(rank)

plt.hist(rank_list)
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
plt.show()
