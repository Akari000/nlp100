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
        pos = []
        dst_surfaces = []
        surfaces = []
        dst_morphs = chunks[chunk.dst].morphs

        for morph in chunk.morphs:  # 係り先
            if morph.pos == '記号':
                continue
            pos.append(morph.pos)
            surfaces.append(morph.surface)
        if '名詞' not in pos:
            continue

        for morph in dst_morphs:    # 係り元
            if morph.pos == '記号':
                continue
            dst_pos.append(morph.pos)
            dst_surfaces.append(morph.surface)
        if '動詞' in dst_pos:
            print('%s\t' %
                  ('').join(surfaces),
                  ('').join(dst_surfaces))

'''
吾輩は   見た
ここで   始めて
ものを   見た
'''
