# 28. MediaWikiマークアップの除去
# 27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ．

import json
import re
from pprint import pprint
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
text[-1] = text[-1].split("\n\nグレート")[0]     # 後ろのいらない行を削除

for line in text:
    line = line.split(' = ')
    info[line[0]] = line[1]

pprint(info)
