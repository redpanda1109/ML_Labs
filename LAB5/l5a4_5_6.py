import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from l5a3 import data_split
from l5a1 import missing_values, label_encoding, one_hot_encoding

p=pd.read_excel('thyroid_dataset.xlsx')
X = p.drop(columns=['Condition', 'Record ID'])
Y = p['Condition']
numerical=["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
categorical = ["sex","on thyroxine","query on thyroxine","on antithyroid medication","sick","pregnant","thyroid surgery","I131 treatment","query hypothyroid","query hyperthyroid","lithium","goitre","tumor","hypopituitary","psych","TSH measured","T3 measured","TT4 measured","T4U measured","FTI measured","TBG measured","referral source"]
ordinal = categorical.copy()
ordinal.remove("referral source")
nominal=["referral source"]

#copied from a1 problem...due to the missing values in dataset
X_train, X_test, y_train, y_test = data_split(X,Y)
X_train, X_test = missing_values(X_train, X_test, numerical, categorical)
X_train=label_encoding(X_train, ordinal)
X_train=one_hot_encoding(X_train, nominal)
X_test=label_encoding(X_test, ordinal)
X_test=one_hot_encoding(X_test, nominal)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0) #so test test has same columns as train, no means fill 0

# a4 qs
neigh = KNeighborsClassifier(n_neighbors=3)     #its creating knn with k=3
neigh.fit(X_train, y_train)     #keeping the info assigned/classified already for test dataset
# a5 qs
accuracy = neigh.score(X_test, y_test)  #predicts the class and compares with actual in test data....accuracy=correct predicted/total predicted
print("Accuracy: ",accuracy)
# a6 qs
y_predictions = neigh.predict(X_test)   #gives the predicted classes for test data
classes=set(y_predictions)
for c in classes:
    print(c,": ",list(y_predictions).count(c))  #prints the final count of classified points
