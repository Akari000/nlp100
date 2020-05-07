# 44. 係り受け木の可視化
# 与えられた文の係り受け木を有向グラフとして可視化せよ．
# 可視化には，係り受け木をDOT言語に変換し，Graphvizを用いるとよい．
# また，Pythonから有向グラフを直接的に可視化するには，pydotを使うとよい．

import pydot
from IPython.display import Image
from nlp41 import get_chunks

graph = pydot.Dot(graph_type='digraph')
chunks = ''

with open('../data/neko.txt.cabocha', 'r') as f:
    text = f.read()
    chunks = get_chunks(text)[7]

for chunk in chunks:
    if chunk.dst == -1:
        continue
    surface = chunk.get_surface()
    dst_surface = chunks[chunk.dst].get_surface()
    edge = pydot.Edge(surface, dst_surface)
    graph.add_edge(edge)

graph.write('../results/nlp44.png', format="png")
Image(graph.create(format='png'))
