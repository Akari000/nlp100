'''66. 検索件数の取得
MongoDBのインタラクティブシェルを用いて，活動場所が「Japan」となっているアーティスト数を求めよ．
'''
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['nlp100_database']
collection = db['nlp100_collection']

# TODO Collection.count_documents を調べる
# warning が出ているのは時間がかかるから．
count = collection.count_documents({'area': 'Japan'})
print(count)

'''
22821
'''

'''shell
$ db.nlp100_collection.find(filter={'area': 'Japan'}).count()
22821
'''
