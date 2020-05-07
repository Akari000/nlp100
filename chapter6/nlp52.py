'''52. ステミング
51の出力を入力として受け取り，Porterのステミングアルゴリズムを適用し，単語と語幹をタブ区切り形式で出力せよ．
Pythonでは，Porterのステミングアルゴリズムの実装としてstemmingモジュールを利用するとよい．
'''

from stemming.porter2 import stem
from nlp50 import get_sentences
from nlp51 import tokenize


if __name__ == "__main__":
    sentences = get_sentences()
    for sentence in sentences[:10]:
        token = tokenize(sentence)
        for word in token:
            print('%s\t%s' % (word, stem(word)))


'''
Natural	Natur
language	languag
processing
From	processing
From
Wikipedia	Wikipedia
the	the
free	free
encyclopedia

Natural	encyclopedia

Natur
language	languag
processing	process
(NLP)	(NLP)
is	is
a	a
field	field
of	of
computer	comput
science	scienc
artificial	artifici
intelligence	intellig
'''
