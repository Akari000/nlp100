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


get_tags('Queen')

'''
kamen rider w 1
related-akb48 1
hard rock 2
70s 1
queen family 1
90s 1
80s 1
glam rock 1
british 4
english 1
uk 2
pop/rock 1
pop-rock 1
britannique 1
classic pop and rock 1
queen 1
united kingdom 1
langham 1 studio bbc 1
kind of magic 1
band 1
rock 6
platinum 1
kamen rider w 1
related-akb48 1
hard rock 2
70s 1
queen family 1
90s 1
80s 1
glam rock 1
british 4
english 1
uk 2
pop/rock 1
pop-rock 1
britannique 1
classic pop and rock 1
queen 1
united kingdom 1
langham 1 studio bbc 1
kind of magic 1
band 1
rock 6
platinum 1
kamen rider w 1
related-akb48 1
hard rock 2
70s 1
queen family 1
90s 1
80s 1
glam rock 1
british 4
english 1
uk 2
pop/rock 1
pop-rock 1
britannique 1
classic pop and rock 1
queen 1
united kingdom 1
langham 1 studio bbc 1
kind of magic 1
band 1
rock 6
platinum 1
'''
