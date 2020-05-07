'''60. KVSの構築
Key-Value-Store (KVS) を用い，アーティスト名（name）から活動場所（area）を検索するためのデータベースを構築せよ．
'''

import gzip
import json
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

with gzip.open('../data/artist.json.gz', 'rt') as f:
    for line in f:
        data = json.loads(line)
        r.set(data['name'], data.get('area', ''))
