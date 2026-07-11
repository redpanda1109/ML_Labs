import random

def mean(lst):
    return sum(lst)/len(lst)

def median(lst):
    lst.sort()
    if len(lst)%2==0:
        return (lst[len(lst)//2-1]+lst[len(lst)//2])/2
    else:
        return lst[len(lst)//2]
    
def mode(lst):
    count=[0]*(max(lst)+1)
    for i in lst:
        count[i]+=1
    return count.index(max(count))


lst=[]
for i in range(25):
    lst.append(random.randint(1,10))
print("List:", lst)
print("Mean:", mean(lst))
print("Median:", median(lst))
print("Mode:", mode(lst))
