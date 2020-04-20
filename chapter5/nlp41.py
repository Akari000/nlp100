# 41. 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ．このクラスは形態素（Morphオブジェクト）のリスト（morphs），
# 係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストのCaboChaの解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，8文目の文節の文字列と係り先を表示せよ．
# 第5章の残りの問題では，ここで作ったプログラムを活用せよ．


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


def get_morphs(text):
    morphs = []
    for sentence in text.split('EOS\n')[:6]:
        if len(sentence) < 1:
            continue
        for line in sentence.split('\n')[:-1]:  # 空文字を除く
            if line[0] == '*':
                continue
            surface = line.split('\t')[0]
            line = line.split('\t')[1].split(',')
            morph = Morph(surface, line[6], line[0], line[1])
            morphs.append(morph)
    return morphs


with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()
morphs = get_morphs(text)
