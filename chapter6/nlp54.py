import re
with open('../data/nlp.txt.xml', 'r') as f:
    text = f.read()

tokens = re.findall(r'<token id="\d+">([\s\S]*?)</token>', text)
for token in tokens[:10]:
    pattern = r'<word>(.*?)</word>\s*?'\
              + r'<lemma>(.*?)</lemma>[\s\S]*?'\
              + r'<POS>(.*?)</POS>'
    data = re.findall(pattern, token)[0]
    print('%s\t%s\t%s' % (data[0], data[1], data[0]))

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
