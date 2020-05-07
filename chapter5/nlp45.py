'''45. 動詞の格パターンの抽出
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい．
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ．
ただし，出力は以下の仕様を満たすようにせよ．

動詞を含む文節において，最左の動詞の基本形を述語とする
述語に係る助詞を格とする
述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
'''

from nlp41 import get_chunks

'''
動詞のインデックスを取得
インデックスで形態素を全て検索．助詞ならverbs['index']にリスト形式でsurfaceを保存．
'''
chunks = ''
with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()
    chunks = get_chunks(text)[7]

# TODO UNIXコマンドで確認する

for ind, chunk in enumerate(chunks):
    particles = []
    if not chunk.has_pos('動詞'):
        continue

    for index in chunk.srcs:
        dst_chunk = chunks[index]
        particles += [morph.surface for morph in dst_chunk.morphs
                      if morph.pos == '助詞']
    if len(particles) < 1:
        continue
    verb = chunk.get_pos('動詞')[0]
    particles = (' ').join(particles)
    print('%s\t%s' %
          (verb.base, particles))


'''
始める  で
見る    は を
'''
