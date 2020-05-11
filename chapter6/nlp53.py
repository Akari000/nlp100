'''53. Tokenization
Stanford Core NLPを用い，入力テキストの解析結果をXML形式で得よ．
また，このXMLファイルを読み込み，入力テキストを1行1単語の形式で出力せよ．
'''
import re
with open('../data/nlp.txt.xml', 'r') as f:
    text = f.read()

tokens = re.findall(r'<token id="\d+">([\s\S]*?)</token>', text)
for token in tokens[:10]:
    print(re.sub(r'\s{2,}', '', token))

'''
<word>Natural</word><lemma>natural</lemma><CharacterOffsetBegin>0</CharacterOffsetBegin><CharacterOffsetEnd>7</CharacterOffsetEnd><POS>JJ</POS><NER>O</NER>
<word>language</word><lemma>language</lemma><CharacterOffsetBegin>8</CharacterOffsetBegin><CharacterOffsetEnd>16</CharacterOffsetEnd><POS>NN</POS><NER>O</NER>
<word>processing</word><lemma>processing</lemma><CharacterOffsetBegin>17</CharacterOffsetBegin><CharacterOffsetEnd>27</CharacterOffsetEnd><POS>NN</POS><NER>O</NER>
<word>From</word><lemma>from</lemma><CharacterOffsetBegin>28</CharacterOffsetBegin><CharacterOffsetEnd>32</CharacterOffsetEnd><POS>IN</POS><NER>O</NER>
<word>Wikipedia</word><lemma>Wikipedia</lemma><CharacterOffsetBegin>33</CharacterOffsetBegin><CharacterOffsetEnd>42</CharacterOffsetEnd><POS>NNP</POS><NER>ORGANIZATION</NER>
<word>,</word><lemma>,</lemma><CharacterOffsetBegin>42</CharacterOffsetBegin><CharacterOffsetEnd>43</CharacterOffsetEnd><POS>,</POS><NER>O</NER>
<word>the</word><lemma>the</lemma><CharacterOffsetBegin>44</CharacterOffsetBegin><CharacterOffsetEnd>47</CharacterOffsetEnd><POS>DT</POS><NER>O</NER>
<word>free</word><lemma>free</lemma><CharacterOffsetBegin>48</CharacterOffsetBegin><CharacterOffsetEnd>52</CharacterOffsetEnd><POS>JJ</POS><NER>O</NER>
<word>encyclopedia</word><lemma>encyclopedia</lemma><CharacterOffsetBegin>53</CharacterOffsetBegin><CharacterOffsetEnd>65</CharacterOffsetEnd><POS>NN</POS><NER>O</NER>
<word>Natural</word><lemma>natural</lemma><CharacterOffsetBegin>67</CharacterOffsetBegin><CharacterOffsetEnd>74</CharacterOffsetEnd><POS>JJ</POS><NER>MISC</NER>
'''