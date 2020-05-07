'''61. KVSの検索
60で構築したデータベースを用い，特定の（指定された）アーティストの活動場所を取得せよ．
'''

import redis

r = redis.Redis(host='localhost', port=6379, db=0)
area = r.get('Sweety')
print(area.decode())

'''
Japan
'''
