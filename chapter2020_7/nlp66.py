'''66. WordSimilarity-353での評価
The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
単語ベクトルにより計算される類似度のランキングと，
人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．
'''
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd

googlenews = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)


def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_similarity(row):
    return cos_similarity(googlenews[row['Word 1']], googlenews[row['Word 2']])


def spearman(x, y):
    x = np.array(x)
    y = np.array(y)
    N = len(x)
    return 1 - (6*sum((x - y)**2) / (N*(N**2 - 1)))


word_sim = pd.read_csv('../data/wordsim353/combined.csv')
word_sim['Vector'] = word_sim.apply(get_similarity, axis=1)

# TODO リストの0番目にはindex0のランクが来るようにする．

human_rank = word_sim.sort_values('Human (mean)').index
vector_rank = word_sim.sort_values('Vector').index
spearman_coefficient = spearman(list(human_rank), list(vector_rank))

print(spearman_coefficient)
'''
0.0653429551674618
'''
