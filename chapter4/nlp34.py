# 34. 「AのB」
# 2つの名詞が「の」で連結されている名詞句を抽出せよ．

import pickle

morpheme_list = []

with open('../data/neko.txt.mecab.list', 'rb') as f:
    morpheme_list = pickle.load(f)

for i, line in enumerate(morpheme_list[1:-1], 1):
    if line['surface'] == 'の' \
       and morpheme_list[i-1]['pos'] == '名詞' \
       and morpheme_list[i+1]['pos'] == '名詞':

        print("%s%s%s" % (
              morpheme_list[i-1]['surface'],
              line['surface'],
              morpheme_list[i+1]['surface']))

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
