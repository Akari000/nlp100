# 29. 国旗画像のURLを取得する
# テンプレートの内容を利用し，国旗画像のURLを取得せよ．（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）

import requests
import json
import re
info = {}

with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

text = text.replace("'", '')                # 強調マークアップの除去
text = text.replace('[', '')                # 内部リンクマークアップの除去
text = text.replace(']', '')
text = re.sub(r'<!-- \S+ -->', '', text)    # コメントアウトの除去
text = re.sub(r'[{}:;#*]', '', text)    # その他のマークアップの除去

text = text.split('==')[0]                  # セクションを除去
text = text.split('\n|')[1:]                # 基礎情報より前の行を除去
text[-1] = text[-1].split("\n\n")[0]     # 後ろのいらない行を削除

for line in text:
    line = line.split(' = ')
    info[line[0]] = line[1]

titles = 'file:' + info['国旗画像']

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


"""
https://upload.wikimedia.org/wikipedia/en/a/ae/Flag_of_the_United_Kingdom.svg
"""
