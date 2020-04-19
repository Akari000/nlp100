# 夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をMeCabを使って形態素解析し，その結果をneko.txt.mecabというファイルに保存せよ．このファイルを用いて，以下の問に対応するプログラムを実装せよ．
# なお，問題37, 38, 39はmatplotlibもしくはGnuplotを用いるとよい．
# 30. 形態素解析結果の読み込み
# 形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

import MeCab
import pickle

mt = MeCab.Tagger()
mt.parse('')
text = ''
morpheme_list = []

with open('../data/neko.txt', 'r') as f:
    text = f.read()

node = mt.parseToNode(text)
while node:
    data = {}
    data['surface'] = node.surface
    data['base'] = node.feature.split(',')[6]
    data['pos'] = node.feature.split(',')[0]
    data['pos1'] = node.feature.split(',')[1]
    morpheme_list.append(data)
    node = node.next


with open('../data/neko.txt.mecab', 'wb') as f:
    pickle.dump(morpheme_list, f)
