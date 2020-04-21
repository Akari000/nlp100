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
        for morph in chunk.morphs:              # 係元
            print(morph.surface, end='')
        print('\t', end='')
        for morph in chunks[chunk.dst].morphs:  # 係先
            if morph.pos == '記号':
                break
            print(morph.surface, end='')
        print()


'''
この    書生というのは
書生というのは  話である
時々    捕えて
我々を  捕えて
捕えて  煮て
煮て    食うという
食うという      話である
'''
