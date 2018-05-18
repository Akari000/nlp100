#delete
str="I am students"
print(str.strip("st"))

#reprace1
print str.replace('student','banana')
#reprace2
import re
print re.sub("a", "A",str)


#list dictionary
dict = {1:"neko", 2:"inu"}
print(dict[1])
dict[3] = "taro"
print(dict)
