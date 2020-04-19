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

count = [line[1] for line in c.most_common()]
plt.hist(count)
plt.gca().set_yscale("log")     # 出現頻度が低い単語が多いため縦軸を対数スケールにする
plt.show()
