# 43. 名詞を含む文節が動詞を含む文節に係るものを抽出
# 名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．ただし，句読点などの記号は出力しないようにせよ．

from nlp41 import get_chunks

with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()
    data = get_chunks(text)
    chunks = data[7]
    for chunk in chunks:
        if chunk.dst == -1:
            continue
        dst_pos = []
        srcs_pos = []
        dst_surface = []
        srcs_surface = []
        dst_morphs = chunks[chunk.dst].morphs

        for morph in chunk.morphs:
            if morph.pos == '記号':
                continue
            srcs_pos.append(morph.pos)
            srcs_surface.append(morph.surface)
        if '名詞' not in srcs_pos:
            continue

        for morph in dst_morphs:
            if morph.pos == '記号':
                continue
            dst_pos.append(morph.pos)
            dst_surface.append(morph.surface)
        if '動詞' in dst_pos:
            print('%s\t' %
                  ('').join(srcs_surface),
                  ('').join(dst_surface))

'''
我々を   捕えて
'''
