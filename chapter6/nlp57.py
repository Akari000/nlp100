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

with open('../data/nlptest.txt.xml', 'r') as f:
    text = f.read()

sentences = {}
for sentence in re.findall(sentence_pattern, text):
    sentence_id = sentence[0]
    sentence = sentence[1]
    deps = re.findall(deps_pattern, sentence)
    if len(deps) < 1:
        continue
    for dep in re.findall(dep_pattern, deps[0]):
        if dep[0] == '0':
            continue
        print(dep)
        child = dep[0]
        child_surface = dep[1]
        parent = dep[2]
        parent_surface = dep[3]

        node = pydot.Node(parent, label=parent_surface)  # add parent node
        graph.add_node(node)
        node = pydot.Node(child, label=child_surface)    # add child node
        graph.add_node(node)
        edge = pydot.Edge(parent, child)
        graph.add_edge(edge)
        graph.write('../results/nlp57-%s.png' % (sentence_id), format="png")
