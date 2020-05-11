'''57. 係り受け解析
Stanford Core NLPの係り受け解析の結果（collapsed-dependencies）を有向グラフとして可視化せよ．
可視化には，係り受け木をDOT言語に変換し，Graphvizを用いるとよい．
また，Pythonから有向グラフを直接的に可視化するには，pydotを使うとよい．
'''

import pydot
import re

sentence_pattern = r'<sentence id="(\d+)">([\s\S]*?)</sentence>'
token_pattern = r'<token id="\d+">\s*?'\
                + r'<word>(.*?)</word>[\s\S]*?</token>'
deps_pattern = r'<dependencies type="basic-dependencies">[\s\S]*?</dependencies>'
dep_pattern = r'\<dep [\s\S]*?'\
              + r'<governor idx="(\d+)">(.*?)</governor>\s*?'\
              + r'<dependent idx="(\d+)">(.*?)</dependent>\s*?'\
              + r'</dep>'
graph = pydot.Dot(graph_type='digraph')
sentences = {}

with open('../data/nlptest.txt.xml', 'r') as f:
    text = f.read()

for sentence in re.findall(sentence_pattern, text):
    sentence_id = sentence[0]
    sentence = sentence[1]
    deps = re.findall(deps_pattern, sentence)
    if len(deps) < 1:
        continue
    for dep in re.findall(dep_pattern, deps[0]):
        if dep[0] == '0':  # ROOTの場合
            continue
        child_id = dep[0]
        child = dep[1]
        parent_id = dep[2]
        parent = dep[3]

        node = pydot.Node(parent_id, label=parent)  # add parent node
        graph.add_node(node)
        node = pydot.Node(child_id, label=child)    # add child node
        graph.add_node(node)
        edge = pydot.Edge(parent_id, child_id)
        graph.add_edge(edge)
        graph.write('../results/nlp57-%s.png' % (sentence_id), format="png")
