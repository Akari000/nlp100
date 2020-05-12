'''68. ソート
"dance"というタグを付与されたアーティストの中でレーティングの投票数が多いアーティスト・トップ10を求めよ．
'''

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['nlp100_database']
collection = db['nlp100_collection']

documents = collection.find(
    filter={'tags.value': 'dance'}).sort([("rating.count", -1)]).limit(10)

for document in documents:
    print('%s\t投票数t:%s' % (document['name'], document['rating']['count']))

'''
Madonna 投票数t:26
Björk   投票数t:23
The Prodigy     投票数t:23
Rihanna 投票数t:15
Britney Spears  投票数t:13
Maroon 5        投票数t:11
Adam Lambert    投票数t:7
Fatboy Slim     投票数t:7
Basement Jaxx   投票数t:6
Cornershop      投票数t:5
'''
