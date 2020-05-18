'''65. MongoDBの検索
MongoDBのインタラクティブシェルを用いて，"Queen"というアーティストに関する情報を取得せよ．
さらに，これと同様の処理を行うプログラムを実装せよ．
'''

from pymongo import MongoClient

# TODO filter= をつけなくてもよい
client = MongoClient('localhost', 27017)
db = client['nlp100_database']
collection = db['nlp100_collection']


for document in collection.find(filter={'name': 'Queen'}):
    print(document)

'''
{'_id': ObjectId('5eb62588e78916f7632892be'), 'area': 'Japan', 'name': 'Queen', 'aliases': [{'name': 'Queen', 'sort_name': 'Queen'}], 'tags': [{'count': 1, 'value': 'kamen rider w'}, {'count': 1, 'value': 'related-akb48'}], 'rating': ''}
{'_id': ObjectId('5eb6259de78916f76329596a'), 'area': 'United Kingdom', 'name': 'Queen', 'aliases': [{'name': '女王', 'sort_name': '女王'}], 'tags': [{'count': 2, 'value': 'hard rock'}, {'count': 1, 'value': '70s'}, {'count': 1, 'value': 'queen family'}, {'count': 1, 'value': '90s'}, {'count': 1, 'value': '80s'}, {'count': 1, 'value': 'glam rock'}, {'count': 4, 'value': 'british'}, {'count': 1, 'value': 'english'}, {'count': 2, 'value': 'uk'}, {'count': 1, 'value': 'pop/rock'}, {'count': 1, 'value': 'pop-rock'}, {'count': 1, 'value': 'britannique'}, {'count': 1, 'value': 'classic pop and rock'}, {'count': 1, 'value': 'queen'}, {'count': 1, 'value': 'united kingdom'}, {'count': 1, 'value': 'langham 1 studio bbc'}, {'count': 1, 'value': 'kind of magic'}, {'count': 1, 'value': 'band'}, {'count': 6, 'value': 'rock'}, {'count': 1, 'value': 'platinum'}], 'rating': {'count': 24, 'value': 92}}
{'_id': ObjectId('5eb625d2e78916f7632b13c2'), 'area': '', 'name': 'Queen', 'aliases': [], 'tags': [], 'rating': ''}
'''

'''shell
$ db.nlp100_collection.find(filter={'name': 'Queen'})
{ "_id" : ObjectId("5eb62588e78916f7632892be"), "area" : "Japan", "name" : "Queen", "aliases" : [ { "name" : "Queen", "sort_name" : "Queen" } ], "tags" : [ { "count" : 1, "value" : "kamen rider w" }, { "count" : 1, "value" : "related-akb48" } ], "rating" : "" }
{ "_id" : ObjectId("5eb6259de78916f76329596a"), "area" : "United Kingdom", "name" : "Queen", "aliases" : [ { "name" : "女王", "sort_name" : "女王" } ], "tags" : [ { "count" : 2, "value" : "hard rock" }, { "count" : 1, "value" : "70s" }, { "count" : 1, "value" : "queen family" }, { "count" : 1, "value" : "90s" }, { "count" : 1, "value" : "80s" }, { "count" : 1, "value" : "glam rock" }, { "count" : 4, "value" : "british" }, { "count" : 1, "value" : "english" }, { "count" : 2, "value" : "uk" }, { "count" : 1, "value" : "pop/rock" }, { "count" : 1, "value" : "pop-rock" }, { "count" : 1, "value" : "britannique" }, { "count" : 1, "value" : "classic pop and rock" }, { "count" : 1, "value" : "queen" }, { "count" : 1, "value" : "united kingdom" }, { "count" : 1, "value" : "langham 1 studio bbc" }, { "count" : 1, "value" : "kind of magic" }, { "count" : 1, "value" : "band" }, { "count" : 6, "value" : "rock" }, { "count" : 1, "value" : "platinum" } ], "rating" : { "count" : 24, "value" : 92 } }
{ "_id" : ObjectId("5eb625d2e78916f7632b13c2"), "area" : "", "name" : "Queen", "aliases" : [ ], "tags" : [ ], "rating" : "" }
'''