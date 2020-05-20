'''69. t-SNEによる可視化
国名に関する単語ベクトルのベクトル空間をt-SNEで可視化せよ．
'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import KeyedVectors


googlenews = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)


def preprocessor(row):
    return row.replace(' ', '_')


def isin_vocab(x):
    return x in googlenews


countries = pd.read_csv('../data/countries.csv')
countries['English'] = countries.English.apply(preprocessor)
countries['isin_vocab'] = countries.English.apply(isin_vocab)
countries = countries[countries.isin_vocab]
vec = countries.English.apply(lambda x: googlenews[x]).values
vec = list(vec)

vec = TSNE(n_components=2).fit_transform(vec)  # ２次元に圧縮
x, y = vec.T

plt.figure(figsize=(12, 12))

for (xi, yi, label) in zip(x, y, countries['English'].values):
    plt.plot(xi, yi, '.')
    plt.annotate(label, xy=(xi, yi))
plt.savefig('../results/nlp2020_69test.png')
