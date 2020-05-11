'''54. 品詞タグ付け
Stanford Core NLPの解析結果XMLを読み込み，単語，レンマ，品詞をタブ区切り形式で出力せよ．
'''
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


'''Stanford Core NLPの解析
$ ./corenlp.sh -annotators tokenize,ssplit,pos,lemma,ner,coref
  -file ~/develop/nlp100/data/nlp.txt
'''

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
