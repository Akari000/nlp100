'''
48. 名詞から根へのパスの抽出
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ． ただし，構文木上のパスは以下の仕様を満たすものとする．

各文節は（表層形の）形態素列で表現する
パスの開始文節から終了文節に至るまで，各文節の表現を"->"で連結する
'''

from nlp41 import get_chunks


def get_path(chunks, target, path):
    if target == -1:
        return ''
    path += ' -> ' + chunks[target].get_surface() + get_path(chunks, chunks[target].dst, '')
    return path


with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()
    data = get_chunks(text)
    chunks = data[7]
    for chunk in chunks:
        if(chunk.has_pos('名詞')):
            path = get_path(chunks, chunk.dst, chunk.get_surface())
            print(path)

'''
吾輩は -> 見た
ここで -> 始めて -> 人間という -> ものを -> 見た
人間という -> ものを -> 見た
ものを -> 見た
'''
