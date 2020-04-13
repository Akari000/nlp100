import gzip
import json
data = {}

with gzip.open('../data/jawiki-country.json.gz', 'rt') as f:
    for line in f:
        data = json.loads(line)
        if data['title'] == 'イギリス':
            break

print(data['text'])


# 21-29用にファイルを保存
with open('../data/jawiki-England.json', "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
