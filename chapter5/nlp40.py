# 40. 係り受け解析結果の読み込み（形態素）
# 形態素を表すクラスMorphを実装せよ．このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
# さらに，CaboChaの解析結果（neko.txt.cabocha）を読み込み，各文をMorphオブジェクトのリストとして表現し，3文目の形態素列を表示せよ．


class Morph(object):
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1


morph_list = []
with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()

for sentence in text.split('EOS\n')[:6]:
    if len(sentence) < 1:
        continue
    for line in sentence.split('\n')[:-1]:  # 空文字を除く
        if line[0] == '*':
            continue
        surface = line.split('\t')[0]
        line = line.split('\t')[1].split(',')
        morph = Morph(surface, line[6], line[0], line[1])
        morph_list.append(morph)


print('surface', morph_list[0].surface)
print('base', morph_list[2].base)
print('pos', morph_list[2].pos)
print('pos1', morph_list[2].pos1)

'''
surface 吾輩
base 吾輩
pos 名詞
pos1 代名詞
'''
