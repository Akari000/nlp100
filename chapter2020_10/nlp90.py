import json
import pickle
import re
import MeCab
import pandas as pd

mt = MeCab.Tagger("-Owakati")
mt.parse('')
'''準備

1. http://www.phontron.com/kftt/index-ja.html ここからデータのみをダウンロード
2. $ tar zxvf kftt-data-1.0.tar.gz 回答
3. kftt-data-1.0/data/orig/ 内のxxx.jaのデータをMecabで形態素解析し，xxx.ja.mecabに保存

'''

DATA_DIR = '../data/kftt-data-1.0/data/orig/'


def delete_brackets(text):
    """
    括弧と括弧内文字列を削除
    """
    table = {
        "(": "（",
        ")": "）"
    }
    for key in table.keys():
        text = text.replace(key, table[key])
    pattern = r'（[^（|^）]*）'
    text = re.sub(pattern, "", text)
    if re.search(pattern, text):
        return delete_brackets(text)
    else:
        return text


def tokenize_ja(text):
    text = mt.parse(text)
    text = re.sub(r" \n", "", text)
    tokens = text.split(' ')
    return tokens


def tokenize_en(text):
    return text.split(' ')


def preprocess_ja(text):
    text = delete_brackets(text)
    text = tokenize_ja(text)
    return text


def preprocess_en(text):
    text = delete_brackets(text)
    text = tokenize_en(text)
    return text


# preprocess
for data_name in ['train', 'test', 'dev']:
    with open('%skyoto-%s.ja' % (DATA_DIR, data_name), 'r', encoding='utf-8') as f:
        lines = f.read()
        ja_lines = lines.split('\n')
        ja_tokens = [preprocess_ja(line) for line in ja_lines]

    # tokenize English
    with open('%skyoto-%s.en' % (DATA_DIR, data_name), 'r', encoding='utf-8') as f:
        lines = f.read()
        en_lines = lines.split('\n')
        en_tokens = [preprocess_en(line) for line in en_lines]

    with open('%skyoto-%s.ja.tokens' % (DATA_DIR, data_name), 'wb') as f:
        pickle.dump(ja_tokens, f)

    with open('%skyoto-%s.en.tokens' % (DATA_DIR, data_name), 'wb') as f:
        pickle.dump(en_tokens, f)


# load
with open('%skyoto-%s.ja.tokens' % (DATA_DIR, 'dev'), 'rb') as f:
    ja_tokens = pickle.load(f)

with open('%skyoto-%s.en.tokens' % (DATA_DIR, 'dev'), 'rb') as f:
    en_tokens = pickle.load(f)

print(ja_tokens[0])
print(en_tokens[0])

'''
['臨済宗', 'は', '、', '中国', '禅', '五', '家', '七', '宗', 'の', 'ひとつ', 'で', '、', '唐', 'の', '臨済', '義', '玄', 'を', '宗祖', 'と', 'する', '。']
['Rinzai', 'Zen', 'Buddhism', 'is', 'one', 'of', 'the', 'Chinese', 'five', 'Houses/seven', 'Schools', 'of', 'Zen', '', 'and', 'Gigen', 'RINZAI', '', 'of', 'Tang', 'was', 'its', 'founder.']
'''
