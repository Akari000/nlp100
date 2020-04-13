text = "I am an NLPer"
w_bigram = []
ch_bigram = []


def ngram(word_list, n, sep=""):
    ngram = []
    for i in range(0, len(word_list)-n+1):
        word = (sep).join(word_list[i:i+n])
        ngram.append(word)
    return ngram


ch_bigram = ngram(text, 2)  # 文字n-gramでは，空白を一文字として扱う
w_bigram = ngram(text.split(" "), 2, " ")

print("character bi_gram\n", ch_bigram)
print("word bi_gram\n", w_bigram)

# character bi_gram
#  ['I ', ' a', 'am', 'm ', ' a', 'an', 'n ', ' N', 'NL', 'LP', 'Pe', 'er']
# word bi_gram
#  ['I am', 'am an', 'an NLPer']
