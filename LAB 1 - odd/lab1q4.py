def checkMax(word):
    letter=max(word, key=word.count)
    count=word.count(letter)
    return letter, count


word=input("Enter a word: ")
if not word.isalpha():
    print("Invalid input")
else:
    letter, count=checkMax(word)
    print("Maximum occuring letter:", letter)
    print("Occurance count: ", count)