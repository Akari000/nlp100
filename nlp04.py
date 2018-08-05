text="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
result={}
text=text.split(' ')
for index,word in enumerate(text,1):
    if index in [1,5,6,7,8,9,15,16,19]:
        result['' + str(index)]=word[0]
    else: result['' + str(index)]=word[:2]

print(result)