# 37. 頻度上位10語
# 出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import pickle
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib

morpheme_list = []
with open('../data/neko.txt.mecab.list', 'rb') as f:
    morpheme_list = pickle.load(f)

words = [node['surface'] for node in morpheme_list]
c = collections.Counter(words)

surface = []
count = []
for line in c.most_common()[:10]:
    surface.append(line[0])
    count.append(line[1])

plt.bar(surface, count)
plt.show()
