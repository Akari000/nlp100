'''64. アナロジーデータでの実験
単語アナロジーの評価データをダウンロードし，
vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
求めた単語と類似度は，各事例の末尾に追記せよ．
'''

from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# TODO :から始まる行をtypeとして各レコードに追加する．

googlenews = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)


columns = ['col%d' % (i)for i in range(4)]
questions_words = pd.read_csv(
    '../data/questions-words.txt', names=columns, sep=' ')
questions_words = questions_words[questions_words.col0 != ':']


def get_most_similar(row):
    most_similar = googlenews.most_similar(
                    positive=[row.col1, row.col2],
                    negative=[row.col0],
                    topn=1)[0]
    return pd.Series(list(most_similar))


questions_words[['most_similar', 'similarity']] = questions_words.progress_apply(get_most_similar, axis=1)
questions_words.to_csv('../data/questions_words.csv',
                       header=False, index=False)
print(questions_words.head(10))

'''
	col0	col1	col2	col3	most_similar	similarity
0	Athens	Greece	Baghdad	Iraq	Baghdad	0.748983
1	Athens	Greece	Bangkok	Thailand	Bangkok	0.743114
2	Athens	Greece	Beijing	China	China	0.718659
3	Athens	Greece	Berlin	Germany	Germany	0.672089
4	Athens	Greece	Bern	Switzerland	Bern	0.690234
5	Athens	Greece	Cairo	Egypt	Egypt	0.762682
6	Athens	Greece	Canberra	Australia	Canberra	0.740721
7	Athens	Greece	Hanoi	Vietnam	Hanoi	0.750990
8	Athens	Greece	Havana	Cuba	Havana	0.726283
9	Athens	Greece	Helsinki	Finland	Helsinki	0.723530
'''
