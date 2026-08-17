import pandas as pd
import numpy as np
from l5a3 import data_split
from l5a1 import missing_values,label_encoding,one_hot_encoding,distance_train,choose_sorting,k_neighbour,k_classes,winning_class

def my_fit(X_train, y_train):
    features=X_train.columns
    train_data=[]
    # each id gives feature vector...so a list of lists(vectors)
    for i in range(len(X_train)):
        s=[]
        for feature in features:
            s.append(X_train.iloc[i][feature])
        train_data.append(s)
    return train_data, y_train

def my_predict(X_test, train_data, y_train, k, sorting_function,features):
    test_data=[]
    # each id gives feature vector
    for i in range(len(X_test)):
        s=[]
        for feature in features:
            s.append(X_test.iloc[i][feature])
        test_data.append(s)
    predict_all=[]
    for data in test_data:
        dist=distance_train(train_data, data)  #all dist with train data & point
        sorted_dist=sorting_function(dist)      #the distances get sorted with tie breaker
        neighbours=k_neighbour(sorted_dist,k)   #take k neighbours
        class_type = k_classes(neighbours, y_train) #find the classes of all k neighbours
        prediction = winning_class(class_type)      #predicts the point's class with majority class rule
        predict_all.append(prediction)
    return predict_all, class_type, neighbours

def my_score(y_test, my_predictions):
    actual_class=list(y_test)
    match=0
    for i in range(len(actual_class)):
        if my_predictions[i] == actual_class[i]:
            match+=1
    accuracy=match/len(actual_class)
    return accuracy



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

def main():
    sorting_function=choose_sorting()   #to choose the sorting function
    k=int(input("\nEnter the value of k: "))    #choose k value

#fit() function
    train_data, y_train = my_fit(X_train, y_train)
    features=X_train.columns

#predict function
    my_predictions, class_type, neighbours=my_predict(X_test, train_data, y_train, k, sorting_function, features)
    classes=set(y_train)
    for c in classes:
        print(c,": ",my_predictions.count(c))  #prints the final count of classified points

#score function
    accuracy=my_score(y_test, my_predictions)
    print("\nAccuracy: ",accuracy)

if __name__ == "__main__":  #i am putting it as def main() cause it is easier to import now...else errors are coming
    main()
