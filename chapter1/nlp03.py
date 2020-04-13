import re
text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
text = re.sub(r"[,.]", "", text)
text = text.split(" ")

count = [len(word) for word in text]
print(count)
