'''
65. アナロジータスクでの正解率
64の実行結果を用い，意味的アナロジー（semantic analogy）と
文法的アナロジー（syntactic analogy）の正解率を測定せよ．
'''
import pandas as pd
from sklearn.metrics import accuracy_score

columns = ['col0', 'col1', 'col2', 'col3', 'most_similar', 'similarity']
questions_words = pd.read_csv('../data/questions_words.csv',
                              names=columns)
'''
3列目... 文法的アナロジー
4列目... 意味的アナロジー
'''

accuracy = accuracy_score(questions_words['col3'], questions_words['most_similar'])
print(accuracy)
'''
0.2018522308636922
'''
