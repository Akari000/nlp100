# 夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をMeCabを使って形態素解析し，その結果をneko.txt.mecabというファイルに保存せよ．このファイルを用いて，以下の問に対応するプログラムを実装せよ．
# なお，問題37, 38, 39はmatplotlibもしくはGnuplotを用いるとよい．

# $ cat neko.txt | mecab -o neko.txt.mecab -E ''

# 30. 形態素解析結果の読み込み
# 形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

import pickle

text = ''
morpheme_list = []

with open('../data/neko.txt.mecab', 'r') as f:
    text = f.readlines()

for line in text:
    data = {}
    data['surface'] = line.split('\t')[0]
    line = line.split('\t')[1].split(',')
    data['base'] = line[6]
    data['pos'] = line[0]
    data['pos1'] = line[1]
    morpheme_list.append(data)


with open('../data/neko.txt.mecab.list', 'wb') as f:
    pickle.dump(morpheme_list, f)
