'''64. MongoDBの構築
アーティスト情報（artist.json.gz）をデータベースに登録せよ．
さらに，次のフィールドでインデックスを作成せよ: name, aliases.name, tags.value, rating.value
'''
from pymongo import MongoClient
import json
import gzip

client = MongoClient('localhost', 27017)
db = client['nlp100_database']
collection = db['nlp100_collection']


with gzip.open('../data/artist.json.gz', 'rt') as f:
    for line in f:
        data = json.loads(line)
        post = {
            "area": data.get('area', ''),
            "name": data['name'],
            "aliases": data.get('aliases', []),
            "tags": data.get('tags', []),
            "rating": data.get('rating', '')}
        collection.insert_one(post)
    collection.create_index([('name', 1)])
    collection.create_index([('aliases.name', 1)])
    collection.create_index([('tags.value', 1)])
    collection.create_index([('rating.value', 1)])


for document in collection.find().limit(10):
    print(document)


'''
{'_id': ObjectId('5eb50eb722d102b52daddce6'), 'name': 'WIK▲N', 'aliases': [], 'tags': ['sillyname'], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddce7'), 'name': 'Gustav Ruppke', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddce8'), 'name': 'Pete Moutso', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddce9'), 'name': 'Zachary', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcea'), 'name': 'The High Level Ranters', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddceb'), 'name': 'The Silhouettes', 'aliases': ['Silhouettes', 'The Sihouettes'], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcec'), 'name': 'Aric Leavitt', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddced'), 'name': 'Fonograff', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcee'), 'name': 'Al Street', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcef'), 'name': 'Love .45', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf0'), 'name': 'Sintellect', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf1'), 'name': 'Evie Tamala', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf2'), 'name': 'Jean-Pierre Martin', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf3'), 'name': 'Deejay One', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf4'), 'name': 'wecamewithbrokenteeth', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf5'), 'name': 'The Blackbelt Band', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf6'), 'name': 'Giant Tomo', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf7'), 'name': 'Decoding Jesus', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf8'), 'name': 'Elvin Jones & Jimmy Garrison Sextet', 'aliases': [], 'tags': [], 'rating': ''}
{'_id': ObjectId('5eb50eb722d102b52daddcf9'), 'name': 'DJ Matthew Grim', 'aliases': ['DJ Matthew Grimm'], 'tags': [], 'rating': ''}
'''