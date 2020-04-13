import random
text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."


for word in text.split(" "):
    if len(word) > 4:
        tmp = word[1:-1]
        tmp = random.sample(tmp, len(tmp))  # random.sample は文字列を扱える
        word = word[0] + ('').join(tmp) + word[-1]
    print(word, end=" ")
