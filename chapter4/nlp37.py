# 37. 頻度上位10語
# 出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import collections
import matplotlib.pyplot as plt
import japanize_matplotlib
from nlp30 import get_morphs
morphs = []

# TODO グラフはファイルに保存する
# TODO グラフにラベルをつける

with open('../data/neko.txt.mecab', 'r') as f:
    lines = f.readlines()
    morphs = get_morphs(lines)

surfaces = [morph['surface'] for morph in morphs]
c = collections.Counter(surfaces)

surface = []
count = []
for line in c.most_common(10):
    surface.append(line[0])
    count.append(line[1])

plt.bar(surface, count)
plt.savfig('../data/nlp37.png')
