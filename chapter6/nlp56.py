'''56. 共参照解析
Stanford Core NLPの共参照解析の結果に基づき
文中の参照表現（mention）を代表参照表現（representative mention）に置換せよ
ただし，置換するときは，「代表参照表現（参照表現）」のように，元の参照表現が分かるように配慮せよ．
'''

import re
with open('../data/nlp.txt.xml', 'r') as f:
    text = f.read()


sentence_pattern = r'<sentence id="(\d+)">([\s\S]*?)</sentence>'
token_pattern = r'<token id="\d+">\s*?'\
                + r'<word>(.*?)</word>[\s\S]*?</token>'
coref_pattern = r'<coreference>([\s\S]*?)</coreference>'

mention_pattern = r'<mention>\s*?'\
                  + r'<sentence>(.*?)</sentence>\s*?'\
                  + r'<start>(.*?)</start>\s*?'\
                  + r'<end>(.*?)</end>[\s\S]*?'\
                  + r'</mention>'
rep_pattern = r'<mention representative="true">\s*?'\
              + r'<sentence>(.*?)</sentence>\s*?'\
              + r'<start>(.*?)</start>\s*?'\
              + r'<end>(.*?)</end>[\s\S]*?'\
              + r'</mention>'


sentences = {}

text = text.replace('-LRB-', '(')
text = text.replace('-RRB-', ')')
for sentence in re.findall(sentence_pattern, text):
    sentence_id = sentence[0]
    sentence = sentence[1]
    tokens = re.findall(token_pattern, sentence)
    sentences[sentence_id] = tokens


for coref in re.findall(coref_pattern, text):
    # 代表参照表現
    data = re.findall(rep_pattern, coref)[0]
    rep_id = data[0]
    start = int(data[1]) - 1
    end = int(data[2]) - 1
    rep = sentences[rep_id][start:end]
    # 参照表現
    for data in reversed(re.findall(mention_pattern, coref)):
        target_id = data[0]
        t_start = int(data[1]) - 1
        t_end = int(data[2]) - 1
        sentence = sentences[target_id]
        sentence.insert(t_end, ')')
        sentence.insert(t_start, '(')
        sentence[t_start:t_start] = rep
        sentences[target_id] = sentence

# 出力
for sentence in sentences.values():
    print(' '.join(sentence))

'''
Natural language processing From Wikipedia , the free encyclopedia Natural language processing ( NLP ) is the free encyclopedia Natural language processing ( NLP ) ( the free encyclopedia Natural language processing ( NLP ) ( a field of computer science ) , artificial intelligence , and linguistics concerned ) with the interactions between computers and human ( natural ) languages .
As such , NLP is related to the area of humani-computer interaction .
Many challenges in NLP involve natural language understanding , that is , enabling ( ( computers ) to derive meaning from human or natural language input , and others involve natural language generation .
History The history of NLP generally starts in the 1950s , although work can be found from earlier periods .
In 1950 , Alan Turing published an article titled `` Computing Machinery and Intelligence '' which proposed what is now called the Alan Turing ( Turing ) test as a criterion of intelligence .
The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English .
The authors claimed that within three or five years , machine translation would be a solved problem .
However , real progress was much slower , and after the ALPAC report in 1966 , which found that ten year long research had failed to fulfill the expectations , funding for machine translation was dramatically reduced .
Little further research in machine translation was conducted until the late 1980s , when the first statistical machine translation systems were developed .
Some notably successful NLP systems developed in the 1960s were SHRDLU , a natural language system working in restricted `` blocks worlds '' with restricted vocabularies , and ELIZA , a simulation of a Rogerian psychotherapist , written by Joseph Weizenbaum between 1964 to 1966 .
Using almost no information about human thought or emotion , ELIZA sometimes provided a startlingly human-like interaction .
When the `` patient '' exceeded the very small knowledge base , ELIZA might provide a generic response , for example , responding to `` My head hurts '' with `` Why do you say your head hurts ? ''
'''
