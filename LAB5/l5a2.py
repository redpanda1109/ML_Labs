# same code as knn
#just change the way you tke the majority class with weights
# weight=1/dist
#max total weight per class will become the final majority class

import numpy as np
import pandas as pd
import math
from l5a3 import data_split

def missing_values(X_train, X_test, numerical, categorical):
    X_train=X_train.replace("?", np.nan)
    X_test=X_test.replace("?", np.nan)
    for col in numerical:
        mean1=X_train[col].mean() #takes the mean 
        X_train[col] = X_train[col].fillna(mean1)
        mean2=X_test[col].mean()
        X_test[col] = X_test[col].fillna(mean2)   #separate for test and train
    #for categorical
    for col in categorical:
        mode1 = X_train[col].mode()[0] #take the most frequest value (0)
        X_train[col] = X_train[col].fillna(mode1)
        mode2 = X_test[col].mode()[0]
        X_test[col] = X_test[col].fillna(mode2)
    return X_train, X_test

def label_encoding(p,ordinal):
    # this type of label encoding so that the same label is assigned in train & test dataset
    for col in ordinal:
        if col=="sex":
            p[col]=p[col].replace({"M":0, "F":1}) 
        else:
            p[col]=p[col].replace({"f":0, "t":1})
    return p

def one_hot_encoding(p,nominal):
    for data in nominal:
        one_hot=pd.get_dummies(p[data], dtype=int)
        p=pd.concat([p, one_hot], axis=1)
        p=p.drop(data, axis=1)
    return p

def distance(train_data, point):
    # euclidean distance
    a=list(train_data)
    sum=0
    for i in range(len(a)):
        sum=sum+((a[i]-point[i])**2)
    sum=math.sqrt(sum)
    return sum

def distance_train(train_data, point):      # calculates the dist btw test point & all train points...returns them in list[tuple(dist,train point)]
    dist=[]
    for i in range(len(train_data)):
        temp=distance(train_data[i], point)
        dist.append((temp,i))
    return dist     #a list of dist is returned which will get sorted soon

def heap_sort(arr):
    a = arr.copy()
    def heapify(n, i):
        largest = i
        l = 2*i + 1
        r = 2*i + 2
        if l < n and a[l] > a[largest]:     #tie breaker fixed here
            largest = l
        if r < n and a[r] > a[largest]:     #tie breaker fixed here
            largest = r
        if largest != i:
            a[i], a[largest] = a[largest], a[i]
            heapify(n, largest)

    n = len(a)
    for i in range(n//2 - 1, -1, -1):
        heapify(n, i)
    for i in range(n-1, 0, -1):
        a[i], a[0] = a[0], a[i]
        heapify(i, 0)
    return a

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr)//2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:     #tuples compared & tie breaker fixed here
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    a=arr.copy()
    def partition(low,high):
        pivot=a[high]
        i=low-1
        for j in range(low, high):
            if a[j] <= pivot:       #this does tuple comparison...tie breaker solved here...in tuples it compares element wise
                i += 1
                a[i], a[j] = a[j], a[i]
        a[i+1], a[high] = a[high], a[i+1]
        return i+1
    def quick(low, high):
        if low < high:
            pi = partition(low, high)
            quick(low, pi - 1)
            quick(pi + 1, high)
    quick(0, len(a)-1)
    return a

def choose_sorting():
    #choose any one of the sorting technique
    print("\nChoose sorting algorithm: \n1. Quick Sort \n2. Merge Sort \n3. Heap Sort \n")
    choice=int(input("Enter choice: "))
    match choice:
        case 1:
            return quick_sort
        case 2:
            return merge_sort
        case 3:
            return heap_sort
        case _:
            print("Invalid")
            return None
    
def k_neighbour(sorted_dist, k):        #since distances are already sorted...take the first k points
    neighbours=[]
    for i in range(k):
        neighbours.append(sorted_dist[i])
    return neighbours

def wk_classes(neighbours, y_train):
    class_type=[]       #make a list of the classes in the k neighbours
    for dist, index in neighbours:
        class_type.append(y_train.iloc[index])  #takes the row of every neighbour and finds the respective class in ytrain (since row no. is same)
    return class_type

def winning_class_weighted(class_type, neighbours):
    target_class=[]  #finding the two unique classes
    for c in class_type:
        if c not in target_class:
            target_class.append(c)
    cond=0
    no_cond=0
    for i in range(len(class_type)):
        if class_type[i] == target_class[0]:
            d=neighbours[i][0]
            cond += (1/d)
        else:
            d=neighbours[i][0]
            no_cond += (1/d)
    if cond>no_cond:
        return target_class[0]  #if any one is majority
    elif cond<no_cond:
        return target_class[1]
    else:
        return class_type[0]    #when tie breaker...then take the class with the first smallest dist


p=pd.read_excel('thyroid_dataset.xlsx')
X=p.drop(columns=['Condition', 'Record ID'])
Y=p['Condition']
numerical=["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
categorical = ["sex","on thyroxine","query on thyroxine","on antithyroid medication","sick","pregnant","thyroid surgery","I131 treatment","query hypothyroid","query hyperthyroid","lithium","goitre","tumor","hypopituitary","psych","TSH measured","T3 measured","TT4 measured","T4U measured","FTI measured","TBG measured","referral source"]
ordinal = categorical.copy()
ordinal.remove("referral source")
nominal=["referral source"]

X_train, X_test, y_train, y_test = data_split(X,Y)
X_train, X_test = missing_values(X_train, X_test, numerical, categorical)
X_train=label_encoding(X_train, ordinal)
X_train=one_hot_encoding(X_train, nominal)
X_test=label_encoding(X_test, ordinal)
X_test=one_hot_encoding(X_test, nominal)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0) #so test test has same columns as train, no means fill 0

features=X_train.columns
train_data=[]
# each id gives feature vector...so a list of lists(vectors)
for i in range(len(X_train)):
    s=[]
    for feature in features:
        s.append(X_train.iloc[i][feature])
    train_data.append(s)
test_data=[]
# each id gives feature vector
for i in range(len(X_test)):
    s=[]
    for feature in features:
        s.append(X_test.iloc[i][feature])
    test_data.append(s)

def main():
    sorting_function=choose_sorting()   #to choose the sorting function
    k=int(input("\nEnter the value of k: "))
    predict_all=[]
    for data in test_data:
        dist=distance_train(train_data, data)  #all dist with train data & point
        sorted_dist=sorting_function(dist)      #the distances get sorted with tie breaker
        neighbours=k_neighbour(sorted_dist,k)   #take k neighbours
        class_type = wk_classes(neighbours, y_train) #find the classes of all k neighbours
        prediction = winning_class_weighted(class_type, neighbours)      #predicts the point's class with majority class rule
        predict_all.append(prediction)

    classes=set(y_train)
    for c in classes:
        print(c,": ",predict_all.count(c))  #prints the final count of classified points

if __name__ == "__main__":  #i am putting it as def main() cause it is easier to import now...else errors are coming
    main()
