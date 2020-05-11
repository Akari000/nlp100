'''59. S式の解析
Stanford Core NLPの句構造解析の結果（S式）を読み込み，文中のすべての名詞句（NP）を表示せよ．
入れ子になっている名詞句もすべて表示すること．
'''

import re

sentence_pattern = r'<sentence id="\d+">([\s\S]*?)</sentence>'
token_pattern = r'<token id="\d+">\s*?'\
                + r'<word>(.*?)</word>[\s\S]*?</token>'
parse_pattern = r'<parse>([\s\S]*?) </parse>'


class Node():
    def __init__(self, childlen, key):
        self.childlen = childlen
        self.key = key

    def search_keys(self):
        if self.child == []:
            return self.key
        return self.get_keys(self.childlen)

    def values(self):
        if isinstance(self.childlen, list):
            for child in self.childlen:
                child.values()
        else:
            print(self.childlen, end=' ')

    def search_key(self, key):
        if self.key == key:
            self.values()
            print()
        if isinstance(self.childlen, list):
            for child in self.childlen:
                child.search_key(key)


def get_key(text):
    # 形は1パターン
    # (key value)
    value = ''
    text = re.findall(r'(.*?) (.*)', text[1:-1])[0]
    key = text[0]
    value = text[1]
    return key, value


def get_childlen(text):
    # 形は2パターン
    # 1. (child)(child)
    # 2. value

    # 子がいないとき
    if text[0] != '(':
        return text

    # 子がいるとき
    head = 0
    count = 0
    childlen = []
    for index, t in enumerate(text):
        if t == '(':
            count += 1
        elif t == ')':
            count -= 1
        if count == 0:  # 子が閉じたとき
            child = text[head:index+1]
            if index < head:  # 2つ飛ばす
                continue
            key, value = get_key(child)
            c_childlen = get_childlen(value)
            child = Node(c_childlen, key)
            head = index + 2  # space と )を除く
            childlen.append(child)
    return childlen


with open('../data/nlp.txt.xml', 'r') as f:
    text = f.read()

for sentence in re.findall(sentence_pattern, text):
    tokens = re.findall(token_pattern, sentence)
    pares = re.findall(parse_pattern, sentence)[0]

    key, value = get_key(pares)
    childlen = get_childlen(value)
    node = Node(childlen, key)
    node.search_key('NP')


'''
(ROOT (S (PP (NP (JJ Natural) (NN language) (NN processing)) (IN From) (NP (NNP Wikipedia))) (, ,) (NP (NP (DT the) (JJ free) (NN encyclopedia) (JJ Natural) (NN language) (NN processing)) (PRN (-LRB- -LRB-) (NP (NN NLP)) (-RRB- -RRB-))) (VP (VBZ is) (NP (NP (NP (DT a) (NN field)) (PP (IN of) (NP (NN computer) (NN science)))) (, ,) (NP (JJ artificial) (NN intelligence)) (, ,) (CC and) (NP (NP (NNS linguistics)) (VP (VBN concerned) (PP (IN with) (NP (NP (DT the) (NNS interactions)) (PP (IN between) (NP (NP (NNS computers)) (CC and) (NP (JJ human) (-LRB- -LRB-) (JJ natural) (-RRB- -RRB-) (NNS languages)))))))))) (. .)))

Natural language processing
Wikipedia
the free encyclopedia Natural language processing -LRB- NLP -RRB-
the free encyclopedia Natural language processing
NLP
a field of computer science , artificial intelligence , and linguistics concerned with the interactions between computers and human -LRB- natural -RRB- languages
a field of computer science
a field
computer science
artificial intelligence
linguistics concerned with the interactions between computers and human -LRB- natural -RRB- languages
linguistics
the interactions between computers and human -LRB- natural -RRB- languages
the interactions
computers and human -LRB- natural -RRB- languages
computers
human -LRB- natural -RRB- languages
'''
