'''55. 固有表現抽出
入力文中の人名をすべて抜き出せ．
'''
import re
with open('../data/nlp.txt.xml', 'r') as f:
    text = f.read()


class Token():
    def __init__(self, word, ner):
        self.word = word
        self.ner = ner


tokens = []
for token in re.findall(r'<token id="\d+">([\s\S]*?)</token>', text):
    pattern = r'<word>(.*?)</word>[\s\S]*?'\
              + r'<NER>(.*?)</NER>'
    data = re.findall(pattern, token)[0]
    token = Token(data[0], data[1])
    if token.ner == 'PERSON':
        print(token.word)
    tokens.append(token)


'''
Alan
Turing
ELIZA
Joseph
Weizenbaum
Wilensky
Meehan
Lehnert
Carbonell
Lehnert
Jabberwacky
Moore
'''
