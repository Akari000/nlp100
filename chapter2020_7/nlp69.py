'''69. t-SNEによる可視化
国名に関する単語ベクトルのベクトル空間をt-SNEで可視化せよ．
'''
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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

clusters = KMeans(n_clusters=5).fit_predict(vec)
vec = TSNE(n_components=2).fit_transform(vec)  # ２次元に圧縮
x, y = vec.T

plt.figure(figsize=(12, 12))
cluster2color = {
    0: 'blue',
    1: 'green',
    2: 'red',
    3: 'orange',
    4: 'yellow'}

for (xi, yi, label, cluster) in zip(x, y, countries['English'].values, clusters):
    plt.plot(xi, yi, '.', color=cluster2color[cluster])
    plt.annotate(label, xy=(xi, yi))
plt.savefig('../results/nlp2020_69test.png')
