import re
with open('../data/nlp.txt.xml', 'r') as f:
    text = f.read()


class Token():
    def __init__(self, word, lemma, ner):
        self.word = word
        self.lemma = lemma
        self.ner = ner


tokens = []
for token in re.findall(r'<token id="\d+">([\s\S]*?)</token>', text):
    pattern = r'<word>(.*?)</word>\s*?'\
              + r'<lemma>(.*?)</lemma>[\s\S]*?'\
              + r'<NER>(.*?)</NER>'
    data = re.findall(pattern, token)[0]
    token = Token(data[0], data[1], data[2])
    if token.ner == 'PERSON':
        print(token.word)
    tokens.append(token)


'''
Natural	natural	Natural
language	language	language
processing	processing	processing
From	from	From
Wikipedia	Wikipedia	Wikipedia
,	,	,
the	the	the
free	free	free
encyclopedia	encyclopedia	encyclopedia
Natural	natural	Natural
'''
