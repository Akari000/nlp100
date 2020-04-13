def cipher(text):
    result = ''
    for t in text:
        if t.islower():
            result += chr(219 - ord(t))
        else:
            result += t
    return result


print('I am a blue cat.')
print(cipher('I am a blue cat.'))
