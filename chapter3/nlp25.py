# 25. テンプレートの抽出
# 記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．

import json
import re
info = {}
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

text = text.split("==")[0]                  # セクションを除去
text = text.split("\n|")[1:]                # 基礎情報より前の行を除去
text[-1] = text[-1].split("}}\n'''")[0]     # 後ろのいらない行を削除

for line in text:
    line = line.split(" = ")
    info[line[0]] = line[1]

print(info)
