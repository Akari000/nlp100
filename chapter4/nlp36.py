# 36. 単語の出現頻度
# 文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

import collections
from nlp30 import get_morphs
morphs = []

with open('../data/neko.txt.mecab', 'r') as f:
    lines = f.readlines()
    morphs = get_morphs(lines)

surfaces = [morph['surface'] for morph in morphs]
count = collections.Counter(surfaces)

for surface in count.most_common():
    print(surface)


'''
の
。
て
、
は
に
を
と
が
た
で
「
」
も
ない
だ
し
から
ある
な
ん
か
いる
事
へ
する
う
もの
君
です
云う
主人
よう
ね
この
御
ば
人
その
一
そう
何
なる
さ
よ
なら
吾輩
い
ます
じゃ
'''
