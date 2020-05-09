'''63. オブジェクトを値に格納したKVS
KVSを用い，アーティスト名（name）からタグと被タグ数（タグ付けされた回数）のリストを検索するためのデータベースを構築せよ．
さらに，ここで構築したデータベースを用い，アーティスト名からタグと被タグ数を検索せよ
'''

import gzip
import json
import redis

r = redis.Redis(host='localhost', port=6379, db=1)

with gzip.open('../data/artist.json.gz', 'rt') as f:
    for line in f:
        data = json.loads(line)
        if 'tags' not in data:
            continue
        for tag in data['tags']:
            tag = json.dumps(tag)
            r.rpush(data['name'], tag)


def get_tags(name):
    for tag in r.lrange(name, 0, -1):
        tag = json.loads(tag)
        print(tag['value'], tag['count'])


get_tags('Sweety')

'''
mandopop 1
taiwanese 1
duo 1
chinese 1
mandopop 1
taiwanese 1
duo 1
chinese 1
あいどる 1
mandopop 1
taiwanese 1
duo 1
chinese 1
mandopop 1
taiwanese 1
duo 1
chinese 1
'''
