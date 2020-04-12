import random
text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

text = text.split(" ")
shuffled = random.sample(text[1:-1], len(text)-2)
text[1:-1] = shuffled
print((' ').join(text))
