'''67. k-meansクラスタリング
国名に関する単語ベクトルを抽出し，
k-meansクラスタリングをクラスタ数k=5として実行せよ．
'''
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
import pandas as pd

googlenews = KeyedVectors.load_word2vec_format(
    '../data/GoogleNews-vectors-negative300.bin', binary=True)

k = 5


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

countries['cluster'] = KMeans(n_clusters=k).fit_predict(vec)

for cluster in range(k):
    print('cluster', cluster)
    print(countries[countries['cluster'] == cluster][
          ['Japanese', 'English']].head(5), end='\n\n')

'''
cluster 0
   Japanese       English
2    アルジェリア       Algeria
4      アンゴラ        Angola
18      ベニン         Benin
22     ボツワナ      Botswana
26  ブルキナファソ  Burkina_Faso

cluster 1
   Japanese     English
6    アルゼンチン   Argentina
20     ボリビア     Bolivia
23     ブラジル      Brazil
28   カーボベルデ  Cabo_Verde
34       チリ       Chile

cluster 2
    Japanese      English
0    アフガニスタン  Afghanistan
10  アゼルバイジャン   Azerbaijan
12     バーレーン      Bahrain
13   バングラデシュ   Bangladesh
19      ブータン       Bhutan

cluster 3
   Japanese    English
8   オーストラリア  Australia
11      バハマ    Bahamas
14    バルバドス   Barbados
17     ベリーズ     Belize
49     ドミニカ   Dominica

cluster 4
   Japanese  English
1     アルバニア  Albania
3      アンドラ  Andorra
7     アルメニア  Armenia
9    オーストリア  Austria
15    ベラルーシ  Belarus

'''