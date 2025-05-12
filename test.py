import random



def shuffle_numbers(s):
    NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    result = ""
    for i in range(len(s)):
        if s[i] in NUMBERS:
            result += random.choice(NUMBERS)
        else:
            result += s[i]

    return result

string = "1 + 2 = 3"

for i in range(5):
    print(shuffle_numbers(string))