# 34. 「AのB」
# 2つの名詞が「の」で連結されている名詞句を抽出せよ．

from nlp30 import get_morphs
morphs = []

with open('../data/neko.txt.mecab', 'r') as f:
    lines = f.readlines()
    morphs = get_morphs(lines)

for i, morph in enumerate(morphs[1:-1], 1):
    if morph['surface'] == 'の' \
       and morphs[i-1]['pos'] == '名詞' \
       and morphs[i+1]['pos'] == '名詞':

        print("%s%s%s" % (
              morphs[i-1]['surface'],
              morph['surface'],
              morphs[i+1]['surface']))

'''
彼の掌
掌の上
書生の顔
はずの顔
顔の真中
穴の中
書生の掌
掌の裏
何の事
肝心の母親
藁の上
笹原の中
池の前
池の上
一樹の蔭
垣根の穴
隣家の三
時の通路
一刻の猶予
家の内
彼の書生
以外の人間
前の書生
おさんの隙
おさんの三
.
.
.
'''
