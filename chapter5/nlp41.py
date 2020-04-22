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
    for sentence in re.findall(r'(\n[\s\S]*?EOS)', text):
        chunks = []
        for clause in re.findall(
                r'\* (\d*) (-?\d+).*?\n([\s\S]*?)(?=\n\*|\nEOS)', sentence):  #(srcs) (dst) (morphs)
            morphs = []
            for line in re.findall(r'(.*?)\t(.*?)(?:$|\n)', clause[2]):  # tab前から行末または\nまで
                surface = line[0]
                feature = line[1].split(',')
                morph = Morph(surface, feature[6], feature[0], feature[1])
                morphs.append(morph)
            chunk = Chunk(morphs, int(clause[1]), int(clause[0]))
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
dst 1 srcs 0
dst 5 srcs 0
吾輩 吾輩 名詞 代名詞
は は 助詞 係助詞
dst 2 srcs 1
ここ ここ 名詞 代名詞
で で 助詞 格助詞
dst 3 srcs 2
始め 始める 動詞 自立
て て 助詞 接続助詞
dst 4 srcs 3
人間 人間 名詞 一般
という という 助詞 格助詞
dst 5 srcs 4
もの もの 名詞 非自立
を を 助詞 格助詞
dst -1 srcs 5
見 見る 動詞 自立
た た 助動詞 *
。 。 記号 句点
'''
