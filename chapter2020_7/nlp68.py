'''68. Ward法によるクラスタリング
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
さらに，クラスタリング結果をデンドログラムとして可視化せよ．
'''
from scipy.cluster.hierarchy import dendrogram, linkage
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

plt.figure(figsize=(32.0, 24.0))
link = linkage(vec, method='ward')
dendrogram(link,
           labels=countries['English'].values,
           leaf_rotation=90,
           leaf_font_size=10)

plt.savefig('../results/nlp2020_68test.png')
