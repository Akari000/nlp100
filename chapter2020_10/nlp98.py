'''98. ドメイン適応
対話のデータを使う
'''

inputs = []
targets = []
DATA_DIR = '../data/kftt-data-1.0/data/dial/'
with open(DATA_DIR + 'twitter_pair_example.txt') as f:
    for line in f.read():
        line = line.split('\t')
        if len(line) < 1:
            continue
        inputs.append(line[0])
        inputs.append(line[1])

size = len(inputs)
inputs = [inputs[:size*0.8], inputs[size*0.8:size*0.9], inputs[size*0.9:]]
targets = [targets[:size*0.8], targets[size*0.8:size*0.9], targets[size*0.9:]]

for data_name, inp, targ in zip(['train', 'test', 'dev'], inputs, targets):
    with open(DATA_DIR + '%s.inputs' % (data_name)) as f:
        f.write('\n'.join(inp))

    with open(DATA_DIR + '%s.targets' % (data_name)) as f:
        f.write('\n'.join(targ))
