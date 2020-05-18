'''60. KVSの構築
Key-Value-Store (KVS) を用い，アーティスト名（name）から活動場所（area）を検索するためのデータベースを構築せよ．
'''

import gzip
import json
import redis

# TODO 名前が被っている場合上書きされないようにする（idをkeyにする, valueを辞書型にするなど）
r = redis.Redis(host='localhost', port=6379, db=0)

with gzip.open('../data/artist.json.gz', 'rt') as f:
    for line in f:
        data = json.loads(line)
        values = r.get(data['name'])
        if values is None or isinstance(values, dict):
            values = {}
        values[data.get('id')] = data.get('area', '')
        r.set(data['name'], values)
