'''63. 加法構成性によるアナロジー
“Spain”の単語ベクトルから”Madrid”のベクトルを引き，”Athens”のベクトルを足したベクトルを計算し，
そのベクトルと類似度の高い10語とその類似度を出力せよ．
'''
from gensim.models import KeyedVectors

googlenews = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)

most_similar = googlenews.most_similar(
                    positive=['Spain', 'Athens'],
                    negative=['Madrid'],
                    topn=10)

for word, cos_sim in most_similar:
    print('%s\t%f' % (word, cos_sim))

'''
Greece	0.689848
Aristeidis_Grigoriadis	0.560685
Ioannis_Drymonakos	0.555291
Greeks	0.545069
Ioannis_Christou	0.540086
Hrysopiyi_Devetzi	0.524844
Heraklio	0.520776
Athens_Greece	0.516881
Lithuania	0.516687
Iraklion	0.514679
'''
