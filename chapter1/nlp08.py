def cipher(text):
    encrypted = ''
    for t in text:
        if t.islower():
            encrypted += chr(219 - ord(t))
        else:
            encrypted += t
    return encrypted


origin = 'I am a blue cat.'
encrypted = cipher(origin)
decrypted = cipher(encrypted)

print("original: ", origin)
print("encrypted: ", encrypted)
print("decrypted: ", decrypted)
# original:  I am a blue cat.
# encrypted:  I zn z yofv xzg.
# decrypted:  I am a blue cat.
