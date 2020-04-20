# 夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をMeCabを使って形態素解析し，その結果をneko.txt.mecabというファイルに保存せよ．このファイルを用いて，以下の問に対応するプログラムを実装せよ．
# なお，問題37, 38, 39はmatplotlibもしくはGnuplotを用いるとよい．

# $ cat neko.txt | mecab -o neko.txt.mecab -E ''

# 30. 形態素解析結果の読み込み
# 形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

from pprint import pprint


def get_morphs(lines):
    morphs = []
    for line in lines:
        morph = {}
        morph['surface'] = line.split('\t')[0]
        line = line.split('\t')[1].split(',')
        morph['base'] = line[6]
        morph['pos'] = line[0]
        morph['pos1'] = line[1]
        morphs.append(morph)
    return morphs


if __name__ == "__main__":
    with open('../data/neko.txt.mecab', 'r') as f:
        lines = f.readlines()
        morphs = get_morphs(lines)
        pprint(morphs[:10])

'''
[{'base': '一', 'pos': '名詞', 'pos1': '数', 'surface': '一'},
 {'base': '\u3000', 'pos': '記号', 'pos1': '空白', 'surface': '\u3000'},
 {'base': '吾輩', 'pos': '名詞', 'pos1': '代名詞', 'surface': '吾輩'},
 {'base': 'は', 'pos': '助詞', 'pos1': '係助詞', 'surface': 'は'},
 {'base': '猫', 'pos': '名詞', 'pos1': '一般', 'surface': '猫'},
 {'base': 'だ', 'pos': '助動詞', 'pos1': '*', 'surface': 'で'},
 {'base': 'ある', 'pos': '助動詞', 'pos1': '*', 'surface': 'ある'},
 {'base': '。', 'pos': '記号', 'pos1': '句点', 'surface': '。'},
 {'base': '名前', 'pos': '名詞', 'pos1': '一般', 'surface': '名前'},
 {'base': 'は', 'pos': '助詞', 'pos1': '係助詞', 'surface': 'は'}]
'''
