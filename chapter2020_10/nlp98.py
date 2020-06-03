'''98. ドメイン適応
対話のデータを使う
'''
import re
import MeCab
from tqdm import tqdm

mt = MeCab.Tagger("-Owakati")
mt.parse('')


DATA_DIR = '../data/kftt-data-1.0/data/orig/'


def tokenize_ja(text):
    text = mt.parse(text)
    text = re.sub(r' \n', '\n', text)
    return text


inputs = []
targets = []
DATA_DIR = '../data/kftt-data-1.0/data/dial/'
with open(DATA_DIR + 'twitter_pair_example.txt') as f:
    for line in f.readlines():
        line = line.split('\t')
        if len(line) < 2:
            continue
        inputs.append(line[0])
        targets.append(line[1])

size = len(inputs)
inputs = [inputs[:int(size*0.8)], inputs[int(size*0.8):int(size*0.9)], inputs[int(size*0.9)]]
targets = [targets[:int(size*0.8)], targets[int(size*0.8):int(size*0.9)], targets[int(size*0.9)]]

for data_name, data_inputs, data_targets in zip(['train', 'test', 'dev'], inputs, targets):
    with open(DATA_DIR + '%s.inputs' % (data_name), 'w') as f:
        tokens_list = []
        for line in tqdm(data_inputs):
            f.write(tokenize_ja(line))

    with open(DATA_DIR + '%s.targets' % (data_name), 'w') as f:
        for line in tqdm(data_targets):
            f.write(tokenize_ja(line))
