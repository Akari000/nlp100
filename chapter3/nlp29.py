# 29. 国旗画像のURLを取得する
# テンプレートの内容を利用し，国旗画像のURLを取得せよ．（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）

import requests

s = requests.Session()
url = "https://en.wikipedia.org/w/api.php"
params = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "file:Flag_of_the_United_Kingdom.svg".encode(),
    "iiprop": "url"
}

res = s.get(url=url, params=params)
data = res.json()
pages = data["query"]['pages']

for v in pages.values():
    print(v['imageinfo'][0]['url'])
