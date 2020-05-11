'''50. 文区切り
(. or ; or : or ? or !) → 空白文字 → 英大文字というパターンを文の区切りと見なし，
入力された文書を1行1文の形式で出力せよ．
'''
import re
filename = '../data/nlp.txt'

# TODO 記号は除去しない
# TODO 一文目が取れない？


def get_sentences():
    with open(filename, 'r') as f:
        text = f.read()
    pattern = r'(.*?[\.;:\?!])\s+(?=[A-Z])'
    sentences = re.findall(pattern, text)
    return sentences


if __name__ == "__main__":
    sentences = get_sentences()
    for sentence in sentences[:10]:
        print(sentence)


'''
Natural language processing (NLP) is a field of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human (natural) languages.
As such, NLP is related to the area of humani-computer interaction.
Many challenges in NLP involve natural language understanding, that is, enabling computers to derive meaning from human or natural language input, and others involve natural language generation.
The history of NLP generally starts in the 1950s, although work can be found from earlier periods.
In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.
The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English.
The authors claimed that within three or five years, machine translation would be a solved problem.
However, real progress was much slower, and after the ALPAC report in 1966, which found that ten year long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced.
Little further research in machine translation was conducted until the late 1980s, when the first statistical machine translation systems were developed.
Some notably successful NLP systems developed in the 1960s were SHRDLU, a natural language system working in restricted "blocks worlds" with restricted vocabularies, and ELIZA, a simulation of a Rogerian psychotherapist, written by Joseph Weizenbaum between 1964 to 1966.
'''

'''
先読みアサーション
(?=)
'''