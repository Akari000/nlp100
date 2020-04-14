# 27. 内部リンクの除去
# 26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ

import json
from pprint import pprint
info = {}
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

text = text.replace("'", '')                # 強調マークアップの除去
text = text.replace('[', '')                # 内部リンクマークアップの除去
text = text.replace(']', '')
text = text.split("==")[0]                  # セクションを除去
text = text.split("\n|")[1:]                # 基礎情報より前の行を除去
text[-1] = text[-1].split("\n}}\nグレート")[0]     # 後ろのいらない行を削除

for line in text:
    line = line.split(" = ")
    info[line[0]] = line[1]

pprint(info)
