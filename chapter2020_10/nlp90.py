'''90. データの準備
機械翻訳のデータセットをダウンロードせよ．訓練データ，開発データ，評価データを整形し，
必要に応じてトークン化などの前処理を行うこと．
ただし，この段階ではトークンの単位として形態素（日本語）および単語（英語）を採用せよ．
'''

import re
import MeCab
import en_core_web_sm
from tqdm import tqdm

nlp = en_core_web_sm.load()
mt = MeCab.Tagger("-Owakati")
mt.parse('')


DATA_DIR = '../data/kftt-data-1.0/data/orig/'


def tokenize_ja(text):
    text = mt.parse(text)
    text = re.sub(r' \n', '\n', text)
    return text


def tokenize_en(text):
    text = nlp(text)
    # print(text.text)
    text = ' '.join([doc.text for doc in text])
    text = re.sub(r' \n', '\n', text)
    return text


for data_name in ['train', 'test', 'dev']:
    # tokenize
    with open('%skyoto-%s.ja' % (DATA_DIR, data_name), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ja_tokens = []
        for line in tqdm(lines):
            ja_tokens.append(tokenize_ja(line))
        print(ja_tokens[:10])

    with open('%skyoto-%s.en' % (DATA_DIR, data_name), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        en_tokens = []
        for line in tqdm(lines):
            en_tokens.append(tokenize_en(line))
        print(en_tokens[:10])

    # save
    with open('%skyoto-%s.tokens.ja' % (DATA_DIR, data_name), 'w') as f:
        f.writelines(ja_tokens)

    with open('%skyoto-%s.tokens.en' % (DATA_DIR, data_name), 'w') as f:
        f.writelines(en_tokens)


'''
with open('%skyoto-%s.tokens_all.ja' % (DATA_DIR, 'train'), 'r') as f:
    ja_tokens = f.readlines()

with open('%skyoto-%s.tokens_all.en' % (DATA_DIR, 'train'), 'r') as f:
    en_tokens = f.readlines()

with open('%skyoto-%s.tokens.ja' % (DATA_DIR, 'train'), 'w') as f:
    f.writelines(ja_tokens[:100000])

with open('%skyoto-%s.tokens.en' % (DATA_DIR, 'train'), 'w') as f:
    f.writelines(en_tokens[:100000])
'''
