'''46. 動詞の格フレーム情報の抽出
45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
45の仕様に加えて，以下の仕様を満たすようにせよ．

項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる
「吾輩はここで始めて人間というものを見た」という例文（neko.txt.cabochaの8文目）を考える．
この文は「始める」と「見る」の２つの動詞を含み，「始める」に係る文節は「ここで」，「見る」に係る文節は「吾輩は」と「ものを」と解析された場合は，次のような出力になるはずである．

始める  で      ここで
見る    は を   吾輩は ものを
'''

from nlp41 import get_chunks

chunks = ''
with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()
    chunks = get_chunks(text)[7]

for chunk in chunks:
    particles = []
    clauses = []
    if chunk.morphs[0].pos != '動詞':
        continue
    for index in chunk.srcs:
        dst_chunk = chunks[index]
        for morph in dst_chunk.morphs:
            if morph.pos == '助詞':
                particles.append(morph.surface)
                clauses.append(dst_chunk.get_surface())
    if len(particles) < 1:
        continue
    particles = (' ').join(particles)
    clauses = (' ').join(clauses)
    print('%s\t%s\t%s' %
          (chunk.morphs[0].base, particles, clauses))


'''
始める	で	ここで
見る	は を	吾輩は ものを
'''
