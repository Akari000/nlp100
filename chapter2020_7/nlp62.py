from gensim.models import KeyedVectors

googlenews = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)

most_similar = googlenews.similar_by_word('United_States')
for word, cos_sim in most_similar:
    print('%s\t%f' % (word, cos_sim))

'''
Unites_States   0.787725
Untied_States   0.754137
United_Sates    0.740072
U.S.    0.731077
theUnited_States        0.640439
America 0.617841
UnitedStates    0.616731
Europe  0.613299
countries       0.604480
Canada  0.601907
'''
