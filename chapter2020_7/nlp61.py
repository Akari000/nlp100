from gensim.models import KeyedVectors
import numpy as np

googlenews = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)

'''
コサイン類似度
cos(a. b) = a・b / |a||b|

内積...                    a・b = np.dot(a, b)
ベクトルの大きさ（ノルム）...  |a| = np.linalg.nor()
'''


def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


cos_sim = cos_similarity(
    googlenews['United_States'], googlenews['U.S.'])
print(cos_sim)


'''
0.7310775
'''
