# 41. 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ．このクラスは形態素（Morphオブジェクト）のリスト（morphs），
# 係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストのCaboChaの解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，8文目の文節の文字列と係り先を表示せよ．
# 第5章の残りの問題では，ここで作ったプログラムを活用せよ．

import re


class Morph(object):
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1


class Chunk(object):
    def __init__(self, morphs, dst, srcs):
        self.morphs = morphs
        self.dst = dst
        self.srcs = srcs


def get_chunks(text):
    data = []
    for sentence in re.findall(r'(\*[\s\S]*?EOS)', text):
        chunks = []
        for clause in re.findall(
                r'\* (\d*) (-?\d+\w+).*?\n([\s\S]*?)(?=\n\*|\nEOS)', sentence):  #(srcs) (dst) (morphs)
            morphs = []
            for line in re.findall(r'(.*?)\t(.*?)(?:$|\n)', clause[2]):  # tab前から行末または\nまで
                surface = line[0]
                feature = line[1].split(',')
                morph = Morph(surface, feature[0], feature[1], feature[6])
                morphs.append(morph)
            chunk = Chunk(morphs, clause[1], clause[0])
            chunks.append(chunk)
        data.append(chunks)
    return data


if __name__ == "__main__":
    with open('../data/neko.txt.cabocha', 'r') as f:
        text = f.read()
        data = get_chunks(text)
        for chunk in data[7]:
            print('dst', chunk.dst, 'srcs', chunk.srcs)
            for morph in chunk.morphs:
                print(morph.surface, morph.base, morph.pos, morph.pos1)


'''
dst 1D srcs 0
この 連体詞 * この
dst 7D srcs 1
書生 名詞 一般 書生
という 助詞 格助詞 という
の 名詞 非自立 の
は 助詞 係助詞 は
dst 4D srcs 2
時々 副詞 一般 時々
dst 4D srcs 3
我々 名詞 代名詞 我々
を 助詞 格助詞 を
dst 5D srcs 4
捕え 動詞 自立 捕える
て 助詞 接続助詞 て
dst 6D srcs 5
煮 動詞 自立 煮る
て 助詞 接続助詞 て
dst 7D srcs 6
食う 動詞 自立 食う
という 助詞 格助詞 という
dst -1D srcs 7
話 名詞 サ変接続 話
で 助動詞 * だ
ある 助動詞 * ある
。 記号 句点 。
'''
