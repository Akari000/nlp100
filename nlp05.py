text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
import re
text1=text.split(" ")
print(text1)
text2 = re.sub(" ","",text)
w_bigram = []
ch_bigram = []

print(text2)
for i in range(0,len(text2)):
    w_bigram.append(text2[i:i+2])
print("word bi_gram")
print(w_bigram)

for i in range(0,len(text1)-1):
    ch_bigram.append(text1[i] + " " + text1[i+1])
print("character bi_gram")
print(ch_bigram)
