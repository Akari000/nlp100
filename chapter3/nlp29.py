# 29. 国旗画像のURLを取得する
# テンプレートの内容を利用し，国旗画像のURLを取得せよ．（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）

import requests
import json
import re
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

info = {}
text = re.findall(r'{{(基礎情報[\s\S]*\n)}}', text)[0]
text = re.sub(r"'{2,5}", '', text)  # 強調マークアップを除去
fields = re.findall(r'\|(.*?) = ([\s\S]*?)(?=\n\|)', text)

for field in fields:
    value = re.sub(
        r"\[\[(?:.*?\||)(.*?)\]\]", r'\1', field[1])  # 内部リンクマークアップを除去
    value = re.sub(r'<.*?>', '', value)  # htmlタグ，コメントアウトの除去
    value = re.sub(r'\[.*?\]', '', value)  # 外部リンクの除去
    value = re.sub(r'{{lang\|.*?\|(.*?)}}', r'\1', value)  # langの除去 {{lang|.*|value}} -> value
    value = re.sub(r'\*', '', value)    # 箇条書きの除去
    value = re.sub(r'.*?\|(.*?)', r'\1', value)    # フォントサイズの除去
    info[field[0]] = value

# ここから
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
