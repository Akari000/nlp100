'''58. タプルの抽出
Stanford Core NLPの係り受け解析の結果（collapsed-dependencies）に基づき，
「主語 述語 目的語」の組をタブ区切り形式で出力せよ．ただし，主語，述語，目的語の定義は以下を参考にせよ．

述語: nsubj関係とdobj関係の子（dependant）を持つ単語
主語: 述語からnsubj関係にある子（dependent）
目的語: 述語からdobj関係にある子（dependent）
'''

import re


class Dependent():
    def __init__(self, type, gov_id, gov, dep_id, dep):
        self.type = type
        self.gov_id = gov_id
        self.gov = gov
        self.dep_id = dep_id
        self.dep = dep

    def __str__(self):
        return '%s %s:%s %s:%s' % (
            self.type, self.gov_id, self.gov, self.dep_id, self.dep)


sentence_pattern = r'<sentence id="\d+">([\s\S]*?)</sentence>'
token_pattern = r'<token id="\d+">\s*?'\
                + r'<word>(.*?)</word>[\s\S]*?</token>'
deps_pattern = r'<dependencies type="basic-dependencies">[\s\S]*?</dependencies>'
dep_pattern = r'\<dep type="(.*?)">[\s\S]*?'\
              + r'<governor idx="(\d+)">(.*?)</governor>\s*?'\
              + r'<dependent idx="(\d+)">(.*?)</dependent>\s*?'\
              + r'</dep>'


with open('../data/nlp.txt.xml', 'r') as f:
    text = f.read()


for sentence in re.findall(sentence_pattern, text):
    tokens = re.findall(token_pattern, sentence)
    dep = re.findall(deps_pattern, sentence)
    if len(dep) < 1:
        continue
    deps = []
    for dep in re.findall(dep_pattern, dep[0]):
        dep = Dependent(dep[0], dep[1], dep[2], dep[3], dep[4])
        deps.append(dep)
    subjects = []
    objects = []
    for dep in deps:
        if dep.type == 'nsubj':
            subjects.append(dep)
        elif dep.type == 'dobj':
            objects.append(dep)
    for subject in subjects:
        for obj in objects:
            if subject.gov_id == obj.gov_id:
                predicate = subject.gov
                print('%s\t%s\t%s' % (subject.dep, predicate, obj.dep))

'''
understanding	enabling	computers
others	involve	generation
Turing	published	article
experiment	involved	translation
ELIZA	provided	interaction
patient	exceeded	base
ELIZA	provide	response
which	structured	information
underpinnings	discouraged	sort
that	underlies	approach
Some	produced	systems
Part	introduced	use
which	make	decisions
systems	rely	which
that	contains	errors
implementations	involved	coding
algorithms	take	set
Some	produced	systems
which	make	decisions
models	have	advantage
they	express	certainty
Systems	have	advantages
Automatic	make	use
that	make	decisions
'''
