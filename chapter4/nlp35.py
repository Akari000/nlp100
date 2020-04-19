# 35. 名詞の連接
# 名詞の連接（連続して出現する名詞）を最長一致で抽出せよ

import pickle

morpheme_list = []

with open('../data/neko.txt.mecab', 'rb') as f:
    morpheme_list = pickle.load(f)

size = len(morpheme_list)
for i, line in enumerate(morpheme_list[:-1]):
    if line['pos'] == '名詞' and morpheme_list[i+1]['pos'] == '名詞':
        print(line['surface'] + morpheme_list[i+1]['surface'], end='')
        j = i+2
        while(j < size):
            if morpheme_list[j]['pos'] == '名詞':
                print(morpheme_list[j]['surface'], end='')
                j += 1
            else:
                print()
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
