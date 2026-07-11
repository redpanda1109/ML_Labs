
def rangeOf(lst):
    if len(lst) < 3:
        print("Range determination not possible")
    else:
        print("Range: ", max(lst) - min(lst))

lst=list(map(int, input("Enter list:").split()))
rangeOf(lst)