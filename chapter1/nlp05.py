import re

w_bigram = []
ch_bigram = []
text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
words = text.split(" ")
text = re.sub(" ", "", text)

for i in range(0, len(text)-1):
    ch_bigram.append(text[i:i+2])

for i in range(0, len(words)-1):
    w_bigram.append(words[i] + " " + words[i+1])

print("character bi_gram")
print(ch_bigram)
print("word bi_gram")
print(w_bigram)
