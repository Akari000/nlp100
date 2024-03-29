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


human_rank = word_sim.sort_values('Human (mean)').index
vector_rank = word_sim.sort_values('Vector').index
N = len(human_rank)
human_sim_rank = [0]*N
vector_sim_rank = [0]*N

for rank, human_index, vector_index in zip(range(N), human_rank, vector_rank):
    human_sim_rank[human_index] = rank
    vector_sim_rank[vector_index] = rank

spearman_coefficient = spearman(human_sim_rank, vector_sim_rank)
print(spearman_coefficient)
'''
0.6997112576768793
'''
