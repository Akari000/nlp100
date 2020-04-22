# 35. 名詞の連接
# 名詞の連接（連続して出現する名詞）を最長一致で抽出せよ

from nlp30 import get_morphs
morphs = []

# TODO プリントの部分でリストに格納するようにする
with open('../data/neko.txt.mecab', 'r') as f:
    lines = f.readlines()
    morphs = get_morphs(lines)

size = len(morphs)
long_nouns = []
for i, morph in enumerate(morphs[:-1]):
    if morph['pos'] == '名詞' and morphs[i+1]['pos'] == '名詞':
        long_noun = morph['surface'] + morphs[i+1]['surface']
        j = i+2
        while(j < size):
            if morphs[j]['pos'] == '名詞':
                long_noun += morphs[j]['surface']
                j += 1
            else:
                long_nouns.append(long_noun)
                break

'''
人間中
一番獰悪
時妙
一毛
その後猫
一度
ぷうぷうと煙
邸内
三毛
書生以外
四五遍
五遍
この間おさん
三馬
.
.
.
'''
