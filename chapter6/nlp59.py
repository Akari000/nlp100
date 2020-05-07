'''59. S式の解析
Stanford Core NLPの句構造解析の結果（S式）を読み込み，文中のすべての名詞句（NP）を表示せよ．入れ子になっている名詞句もすべて表示すること．
'''

import re

sentence_pattern = r'<sentence id="\d+">([\s\S]*?)</sentence>'
token_pattern = r'<token id="\d+">\s*?'\
                + r'<word>(.*?)</word>[\s\S]*?</token>'
parse_pattern = r'<parse>([\s\S]*?) </parse>'
'''
S式の構造
ROOT, document, sentences, sentence, parse
(ROOT 
    (S 
        (NP 
            (NP 
                (NN History) 
                (DT The) 
                (NN history)
            ) 
            (PP 
                (IN of) 
                (NP 
                    (NN NLP)
                )
            )
        ) 
        (ADVP
            (RB generally)
        ) 
        (VP 
            (VBZ starts) 
            (PP 
                (IN in) 
                (NP 
                    (DT the) 
                    (CD 1950s)
                )
            ) 
            (, ,) 
            (SBAR 
                (IN although) 
                (S 
                    (NP 
                        (NN work)
                    ) 
                    (VP 
                        (MD can) 
                        (VP 
                            (VB be) 
                            (VP 
                                (VBN found) 
                                (PP 
                                    (IN from) 
                                    (NP 
                                        (JJR earlier) 
                                        (NNS periods)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ) 
        (. .)
    )
)
'''
'''
nodeを全て定義する．再帰処理でvalueがNPの子ノードを探す．
見つけたらプリントする．
'''


class Node():
    def __init__(self, child, value):
        self.child = child
        self.value = value

    def get_values(self):
        if self.child == -1:
            return self.value
        return self.get_values(self.child)


def s_paser(text):
    if text[0] != '(' or text[-1] != ')':
        print('parse error', text[0], text[-1])
        return -1

    value = ''
    childlen = []
    child_head = -1
    for index, t in enumerate(text[1:], 1):  # valueの抽出
        if t == '(' or t == ')':
            child_head = index
            break
        value += t

    if child_head == -1:  # 子がいないとき
        return Node(childlen, value)

    count = 0
    for index, t in enumerate(text[child_head:], child_head):
        if t == '(':
            count += 1
        elif t == ')':
            count -= 1
        if count == 0:  # 子が閉じたとき
            child = text[child_head:index]
            child = s_paser(child)
            child_head = index
            childlen.append(child)
    return Node(childlen, value)


with open('../data/nlp.txt.xml', 'r') as f:
    text = f.read()

for sentence in re.findall(sentence_pattern, text):
    tokens = re.findall(token_pattern, sentence)
    pares = re.findall(parse_pattern, sentence)[0]
    pares = s_paser(pares)
    # print(pares.child, end='')
    print('====')
