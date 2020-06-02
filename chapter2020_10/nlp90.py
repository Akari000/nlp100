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
        
    pairs = pd.DataFrame()
    pairs['ja'] = ja_tokens
    pairs['en'] = en_tokens
    pairs['ja_orig'] = ja_lines
    pairs['en_orig'] = en_lines
    pairs.to_csv('%skyoto-%s.pairs' % (DATA_DIR, data_name),
                 header=False, index=False)

columns = ('ja', 'en', 'ja_orig', 'en_orig')
pairs = pd.read_csv('%skyoto-%s.pairs' % (DATA_DIR, 'dev'), names=columns)
print(pairs.head(5))
'''
                                                    ja                                                 en                                            ja_orig                                            en_orig
0  ['臨済宗', 'は', '、', '中国', '禅', '五', '家', '七', '宗...  ['Rinzai', 'Zen', 'Buddhism', 'is', 'one', 'of...  臨済宗（臨濟宗、りんざいしゅう）は、中国禅五家七宗（ごけしちしゅう）（臨済、潙仰宗、曹洞宗、...  Rinzai Zen Buddhism is one of the Chinese five...
1  ['彼', 'は', '『', '喝', 'の', '臨済', '』', '『', '臨済'...  ['He', 'was', 'known', 'as', '"RINZAI', 'of', ...     彼は『喝の臨済』『臨済将軍』の異名で知られ、豪放な家風を特徴として中国禅興隆の頂点を極めた。  He was known as "RINZAI of Katu (meaning to he...
2  ['公案', 'に', '参究', 'する', 'こと', 'により', '見性', 'しよ...  ['With', 'its', 'Zen', 'Talks', 'that', 'try',...  公案に参究することにより見性しようとする看話禅（かんなぜん）で、ただ座禅する曹洞宗の黙照禅と...  With its Zen Talks that try to awaken self awa...
3                              ['中国', 'における', '臨済宗']                ['Rinzai', 'School', 'in', 'China']                                          中国における臨済宗                             Rinzai School in China
4  ['臨済宗', 'は', '、', 'その', '名', 'の', '通り', '、', '...  ['As', 'the', 'name', 'implies,', 'Rinzai', 'S...                  臨済宗は、その名の通り、会昌の廃仏後、唐末の宗祖臨済義玄に始まる。  As the name implies, Rinzai School started wit...
'''
