import pandas as pd
from collections import Counter
DATA_DIR = '../data/kftt-data-1.0/data/orig/'

columns = ('ja', 'en', 'ja_orig', 'en_orig')
train = pd.read_csv('%skyoto-%s.pairs' % (DATA_DIR, 'train'), names=columns)


'''
1. enとjaそれぞれword2idとid2wordを作る : 2回以上出てくる単語のみidを振り，それ以外は2にする．
2. 日本語が10単語以上の文は削除する．
3. tokensをidに変える．文頭と文末に0(sos), 1(eos)を追加する
4. lengsを保存する
5. paddingする
'''


def get_word2id(tokens):  # input: list of tokens
    tokens = sum(tokens, [])  # flat list
    counter = Counter(tokens)
    word2id = {}
    for index, (token, freq) in enumerate(counter.most_common(), 1):
        if freq < 2:
            word2id[token] = 0
        else:
            word2id[token] = index
    return word2id


def encode(tokens, word2id, sos=0, eos=1, unk=2):
    ids = [sos]
    for token in tokens:
        if token in word2id:
            ids.append(word2id[token])
        else:
            ids.append(unk)
    ids.append(eos)
    leng = len(ids)
    return ids, leng


tokens = train['ja'].values.tolist()
word2id_ja = get_word2id(tokens)
X_train_ja = train.ja.apply(encode)

tokens = train['en'].values.tolist()
word2id_en = get_word2id(tokens)
X_train_en = train.en.apply(encode)
