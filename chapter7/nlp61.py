'''61. KVSの検索
60で構築したデータベースを用い，特定の（指定された）アーティストの活動場所を取得せよ．
'''

import redis

r = redis.Redis(host='localhost', port=6379, db=0)


def get_area(name):
    area = r.get(name)
    return area.decode()


print(get_area('Sweety'))

'''
Japan
'''
