import re
text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
w_bigram = []
ch_bigram = []

words = re.sub(" ", "", text)
for i in range(len(text)-1):
    ch_bigram.append(text[i:i+2])

words = text.split(" ")
for i in range(len(words)-1):
    w_bigram.append(words[i] + " " + words[i+1])

print("character bi_gram\n", ch_bigram)
print("word bi_gram\n", w_bigram)
