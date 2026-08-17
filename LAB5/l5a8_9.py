import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from l5a3 import data_split
from l5a1 import missing_values,label_encoding,one_hot_encoding, distance_train,choose_sorting,k_neighbour
from l5a7 import my_fit, my_predict, my_score
from sklearn.neighbors import KNeighborsClassifier 
from l5a2 import wk_classes, winning_class_weighted

def my_knn(X_test, y_train, y_test, k, sorting_function, train_data, features):
    predictions, class_types, neighbours = my_predict(X_test,train_data,y_train,k,sorting_function, features)  #(_) indicates unwanted parameter from the func
    accuracy = my_score(y_test, predictions)
    return accuracy, class_types, neighbours

def my_wknn(y_test, class_types, neighbours, test_data, train_data):
    predict_all=[]
    for data in test_data:
        prediction = winning_class_weighted(class_types, neighbours)      #predicts the point's class with majority class rule
        predict_all.append(prediction)
    accuracy=my_score(y_test, predict_all)
    return accuracy

def package_knn(X_train, X_test, y_train, y_test, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    accuracy = neigh.score(X_test, y_test)
    return accuracy

def plot_accuracy(n, knn_accuracy, wknn_accuracy, package_accuracy):
    plt.plot(n, knn_accuracy, label='My_KNN')
    plt.plot(n, wknn_accuracy, label='My_WKNN')
    plt.plot(n, package_accuracy, label='Package KNN')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("knn_comparison.png")
    plt.show()

    

p=pd.read_excel('thyroid_dataset.xlsx')
X = p.drop(columns=['Condition', 'Record ID'])
Y = p['Condition']
numerical=["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
categorical = ["sex","on thyroxine","query on thyroxine","on antithyroid medication","sick","pregnant","thyroid surgery","I131 treatment","query hypothyroid","query hyperthyroid","lithium","goitre","tumor","hypopituitary","psych","TSH measured","T3 measured","TT4 measured","T4U measured","FTI measured","TBG measured","referral source"]
ordinal = categorical.copy()
ordinal.remove("referral source")
nominal=["referral source"]

X_train, X_test, y_train, y_test = data_split(X, Y)
X_train, X_test = missing_values(X_train, X_test, numerical, categorical)
X_train=label_encoding(X_train, ordinal)
X_train=one_hot_encoding(X_train, nominal)
X_test=label_encoding(X_test, ordinal)
X_test=one_hot_encoding(X_test, nominal)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0) #so test test has same columns as train, no means fill 0

sorting_function = choose_sorting()
knn_accuracy=[]
wknn_accuracy=[]
package_accuracy=[]

train_data, y_train = my_fit(X_train, y_train)
features=X_train.columns
test_data=[]
    # each id gives feature vector
for i in range(len(X_test)):
    s=[]
    for feature in features:
        s.append(X_test.iloc[i][feature])
    test_data.append(s)


for k in range(10,21):
    acc1, class_types, neighbours=my_knn(X_test, y_train, y_test, k, sorting_function, train_data, features)
    knn_accuracy.append(acc1)
    acc2=my_wknn(y_test, class_types, neighbours, test_data, train_data)
    wknn_accuracy.append(acc2)
    acc3=package_knn(X_train, X_test, y_train, y_test, k)
    package_accuracy.append(acc3)

n=list(range(10,21))
plot_accuracy(n,knn_accuracy,wknn_accuracy,package_accuracy)