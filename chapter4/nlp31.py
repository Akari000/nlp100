# 31. 動詞
# 動詞の表層形をすべて抽出せよ．

from nlp30 import get_morphs
morphs = []

with open('../data/neko.txt.mecab', 'r') as f:
    lines = f.readlines()
    morphs = get_morphs(lines)

for morph in morphs:
    if morph['pos'] == '動詞':
        print(morph['surface'])


'''
離れ
離れる
分
れる
なる
いる
思っ
てる
違い
いる
いる
なる
合わ
なら
し
云っ
見える
し
死ん
化ける
行か
し
穿い
鍛え上げ
乗り込ん
くる
思う
なる
なる
なれ
なる
する
すれ
する
合わ
なる
合わ
する
.
.
.
'''
