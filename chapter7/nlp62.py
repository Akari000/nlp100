'''62. KVS内の反復処理
60で構築したデータベースを用い，活動場所が「Japan」となっているアーティスト数を求めよ．
'''

import redis

r = redis.Redis(host='localhost', port=6379, db=0)
artists = []
for key in r.scan_iter():
    area = r.get(key)
    if area == b'Japan':
        artists.append(key.decode())

print('アーティスト数:', len(artists))

'''
アーティスト数: 21946
'''
