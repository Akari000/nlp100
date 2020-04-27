# 42. 係り元と係り先の文節の表示
# 係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．ただし，句読点などの記号は出力しないようにせよ．

from nlp41 import get_chunks

with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()
    data = get_chunks(text)
    chunks = data[7]
    for chunk in chunks:
        if chunk.dst == -1:
            continue
        print('%s\t%s' % (
            chunk.get_surface(),                # 係元
            chunks[chunk.dst].get_surface()     # 係先
        ))


'''
吾輩は  見た
ここで  始めて
始めて  人間という
人間という      ものを
ものを  見た
'''
