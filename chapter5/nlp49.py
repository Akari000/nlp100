'''49. 名詞間の係り受けパスの抽出
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．

問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を"->"で連結して表現する
文節iとjに含まれる名詞句はそれぞれ，XとYに置換する
また，係り受けパスの形状は，以下の2通りが考えられる．

文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を"|"で連結して表示
'''

from nlp41 import get_chunks
import re


def get_path(chunks, i):
    target = chunks[i].dst
    if target == -1:
        return ''
    path = []
    path.append(chunks[target].get_surface())
    path += get_path(chunks, target)
    return path


def get_pire_path(chunks, i, j):
    target = chunks[i].dst
    if target == -1 or i == j-1:
        return ''
    path = []
    path.append(chunks[target].get_surface())
    path += get_pire_path(chunks, target, j)
    return path


def get_min_path(chunks, i, j):
    i_path = get_path(chunks, i)
    # 文節iから構文木の根に至る経路上に文節jが存在する場合
    if chunks[j].get_surface() in i_path:
        path = get_pire_path(chunks, i, j)
        if len(path) > 0:
            path = '-> ' + (' -> ').join(path) + ' ->'
        else:
            path = '->'
        return '%s %s %s' % (
            chunks[i].replace_surface('名詞', 'X'), path, 'Y')
    # 文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合:
    j_path = get_path(chunks, j)
    common = []
    sub = []
    for i in range(len(i_path)):
        common = re.findall((' ').join(i_path[i:]), (' ').join(j_path))
        if len(common) > 0:
            sub = re.sub((' ').join(i_path[i:]), '', (' ').join(j_path))
            sub = sub[:-1].split(' ')
    if len(common) > 0:
        if sub == ['']:
            sub = chunks[j].get_surface()
        else:
            sub = [chunks[j].get_surface()] + sub
            sub = (' -> ').join(sub)
        return '%s | %s | %s' % (
            chunks[i].get_surface(),
            sub,
            ('').join(common))
    return ''


with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()
    data = get_chunks(text)
    chunks = data[7]


for i in range(len(chunks)):
    if(not chunks[i].has_noun()):
        continue
    for j in range(i+1, len(chunks)):
        if(not chunks[j].has_noun()):
            continue
        path = get_min_path(chunks, i, j)
        print(path)


'''
吾輩は | ここで -> 始めて -> 人間という -> ものを | 見た
吾輩は | 人間という -> ものを | 見た
吾輩は | ものを | 見た
ここで -> 始めて -> 人間という
ここで -> 始めて -> 人間という -> ものを
人間という -> ものを
'''