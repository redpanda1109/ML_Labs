def pairs(lst):
    count=0
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i]+lst[j]==10:
                count+=1
    return count

lst=[2, 7, 4, 1, 3, 6]
print("No.of pairs:", pairs(lst))
