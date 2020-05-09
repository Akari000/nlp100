'''67. 複数のドキュメントの取得
特定の（指定した）別名を持つアーティストを検索せよ．
'''

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['nlp100_database']
collection = db['nlp100_collection']


def get_documents(aliase):
    return collection.find(filter={'aliases.name': aliase})


aliase = 'Queen'
documents = get_documents(aliase)
for document in documents:
    print(document)

'''
{'_id': ObjectId('5eb62588e78916f7632892be'), 'area': 'Japan', 'name': 'Queen', 'aliases': [{'name': 'Queen', 'sort_name': 'Queen'}], 'tags': [{'count': 1, 'value': 'kamen rider w'}, {'count': 1, 'value': 'related-akb48'}], 'rating': ''}
'''

'''shell
$ db.nlp100_collection.find(filter={'aliases.name': 'Queen'})
{ "_id" : ObjectId("5eb62588e78916f7632892be"), "area" : "Japan", "name" : "Queen", "aliases" : [ { "name" : "Queen", "sort_name" : "Queen" } ], "tags" : [ { "count" : 1, "value" : "kamen rider w" }, { "count" : 1, "value" : "related-akb48" } ], "rating" : "" }
'''
