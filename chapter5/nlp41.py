# 41. 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ．このクラスは形態素（Morphオブジェクト）のリスト（morphs），
# 係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストのCaboChaの解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，8文目の文節の文字列と係り先を表示せよ．
# 第5章の残りの問題では，ここで作ったプログラムを活用せよ．

import re

# TODO 実例を書く
# TODO srcsを係り元文節インデックス番号のリストにする ex) srcs = [2,3,4]


class Morph(object):
    srcs = -1

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

    def get_surface(self):
        surface = [morph.surface for morph in self.morphs]
        surface = ('').join(surface)
        return re.sub(r'[、。]', '', surface)

    def get_pos(self, pos):
        return [morph for morph in self.morphs
                if morph.pos == pos]

    # 以下は46.から使用
    def apply_index(self, index):
        for morph in self.morphs:
            morph.chunk_id = index

    def has_pos(self, pos):
        return pos in [morph.pos for morph in self.morphs]

    def replace_pos(self, pos, X):
        surface = ''
        for morph in self.morphs:
            if morph.pos == pos:
                surface += X
            else:
                surface += morph.surface
        return re.sub(r'[、。]', '', surface)


def get_chunks(text):
    data = []
    for sentence in re.findall(r'(\n[\s\S]*?EOS)', text):
        chunks = []
        for clause in re.findall(
                r'\* (\d*) (-?\d+).*?\n([\s\S]*?)(?=\n\*|\nEOS)', sentence):  # (index) (dst) (morphs)
            morphs = []
            srcs = re.findall(
                r'\* (\d*) ' + clause[0] + r'D.*?\n[\s\S]*?(?=\n\*|\nEOS)',
                sentence)
            for line in re.findall(r'(.*?)\t(.*?)(?:$|\n)', clause[2]):  # tab前から行末または\nまで
                surface = line[0]
                feature = line[1].split(',')
                morph = Morph(surface, feature[6], feature[0], feature[1])
                morphs.append(morph)
            chunk = Chunk(morphs, int(clause[1]), list(map(int, srcs)))
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
dst 5 srcs []
吾輩 吾輩 名詞 代名詞
は は 助詞 係助詞
dst 2 srcs []
ここ ここ 名詞 代名詞
で で 助詞 格助詞
dst 3 srcs ['1']
始め 始める 動詞 自立
て て 助詞 接続助詞
dst 4 srcs ['2']
人間 人間 名詞 一般
という という 助詞 格助詞
dst 5 srcs ['3']
もの もの 名詞 非自立
を を 助詞 格助詞
dst -1 srcs ['0', '4']
見 見る 動詞 自立
た た 助動詞 *
。 。 記号 句点
'''
