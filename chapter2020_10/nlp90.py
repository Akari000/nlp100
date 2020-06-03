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

    with open('%skyoto-%s.en' % (DATA_DIR, data_name), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        en_tokens = []
        for line in tqdm(lines):
            en_tokens.append(tokenize_en(line))

    # save
    with open('%skyoto-%s.tokens.ja' % (DATA_DIR, data_name), 'w') as f:
        f.writelines(ja_tokens)

    with open('%skyoto-%s.tokens.en' % (DATA_DIR, data_name), 'w') as f:
        f.writelines(en_tokens)


'''
kyoto-train.tokens.ja
雪舟 （ せっしゅう 、 1420 年 （ 応永 27 年 ） - 1506 年 （ 永 正 3 年 ） ） は 号 で 、 15 世紀 後半 室町 時代 に 活躍 し た 水墨 画家 ・ 禅僧 で 、 画聖 と も 称え られる 。
日本 の 水墨 画 を 一変 さ せ た 。
諱 は 「 等 楊 （ とう よう ） 」 、 もしくは 「 拙 宗 （ せっしゅう ） 」 と 号 し た 。

kyoto-train.tokens.en
Known as Sesshu ( 1420 - 1506 ) , he was an ink painter and Zen monk active in the Muromachi period in the latter half of the 15th century , and was called a master painter .
He revolutionized the Japanese ink painting .
He was given the posthumous name " Toyo " or " Sesshu ( 拙宗 ) . "
'''
