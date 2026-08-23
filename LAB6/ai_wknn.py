# AI Tool Used: ChatGPT
# Lab 6 - AI-generated weighted k-NN implementation

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ai_data_split import ai_data_split


# AI Tool Used: ChatGPT
def ai_wknn_missing_values(X_train, X_test, numerical, categorical):
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Convert '?' into missing values
    X_train = X_train.replace("?", np.nan)
    X_test = X_test.replace("?", np.nan)

    # Convert numerical columns to numeric values
    for col in numerical:
        X_train[col] = pd.to_numeric(X_train[col],errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col],errors="coerce")

    # Numerical columns -> training median
    for col in numerical:
        median_value = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)
    # Categorical columns -> training mode
    for col in categorical:
        mode_value = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_value)
        X_test[col] = X_test[col].fillna(mode_value)
    return X_train, X_test


# AI Tool Used: ChatGPT
def ai_wknn_label_encoding(data, binary_columns):
    data = data.copy()
    for col in binary_columns:
        if col == "sex":
            data[col] = data[col].replace({"M": 0,"F": 1})
        else:
            data[col] = data[col].replace({"f": 0,"t": 1})
    return data


# AI Tool Used: ChatGPT
def ai_wknn_one_hot_encoding(X_train,X_test,nominal_columns):
    X_train = X_train.copy()
    X_test = X_test.copy()
    encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    train_encoded = encoder.fit_transform(X_train[nominal_columns])
    test_encoded = encoder.transform(X_test[nominal_columns])
    encoded_columns = encoder.get_feature_names_out(nominal_columns)

    train_encoded = pd.DataFrame(train_encoded,columns=encoded_columns,index=X_train.index)
    test_encoded = pd.DataFrame(test_encoded,columns=encoded_columns,index=X_test.index)

    X_train = X_train.drop(columns=nominal_columns)
    X_test = X_test.drop(columns=nominal_columns)
    X_train = pd.concat([X_train, train_encoded],axis=1)
    X_test = pd.concat([X_test, test_encoded],axis=1)
    return X_train, X_test


# AI Tool Used: ChatGPT
def euclidean_distance(train_vector, test_vector):
    squared_sum = 0.0
    for i in range(len(train_vector)):
        difference = (train_vector[i] - test_vector[i])
        squared_sum += (difference * difference)
    return math.sqrt(squared_sum)


# AI Tool Used: ChatGPT
def ai_wknn_calculate_distances(X_train, test_vector):
    distances = []
    for index in range(len(X_train)):
        distance = euclidean_distance(X_train[index],test_vector)
        distances.append((distance, index))
    return distances


# AI Tool Used: ChatGPT
def ai_wknn_sort_distances(distances):
    if len(distances) <= 1:
        return distances
    pivot = distances[len(distances) // 2]
    left = []
    middle = []
    right = []

    for pair in distances:
        if pair[0] < pivot[0]:
            left.append(pair)
        elif pair[0] > pivot[0]:
            right.append(pair)
        else:
            # Equal distances, use training index as tie breaker
            if pair[1] < pivot[1]:
                left.append(pair)
            elif pair[1] > pivot[1]:
                right.append(pair)
            else:
                middle.append(pair)

    return (ai_wknn_sort_distances(left)+ middle+ ai_wknn_sort_distances(right))


# AI Tool Used: ChatGPT
def ai_wknn_neighbors(sorted_distances, k):
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if k > len(sorted_distances):
        raise ValueError("k cannot be greater than the number of training samples.")
    return sorted_distances[:k]


# AI Tool Used: ChatGPT
def ai_class_weights(neighbors, y_train):
    class_weights = {}
    for distance, index in neighbors:
        class_label = y_train[index]
        if distance == 0:
            weight = float("inf")
        else:
            weight = 1.0 / distance
        if class_label not in class_weights:
            class_weights[class_label] = 0.0
        class_weights[class_label] += weight
    return class_weights


# AI Tool Used: ChatGPT
def ai_weighted_majority(neighbors,y_train,class_weights):
    maximum_weight = max(class_weights.values())
    highest_weight_classes = {class_label for class_label, weight in class_weights.items() if weight == maximum_weight}
    # No tie
    if len(highest_weight_classes) == 1:
        return next(iter(highest_weight_classes))

    # Tie breaker, closest neighbour
    for distance, index in neighbors:
        class_label = y_train[index]
        if class_label in highest_weight_classes:
            return class_label


# AI Tool Used: ChatGPT
def ai_wknn_predict(X_train,y_train,X_test,k):
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of samples.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if k > len(X_train):
        raise ValueError("k cannot be greater than the number of training samples.")

    predictions = []
    for test_vector in X_test:
        distances = ai_wknn_calculate_distances(X_train,test_vector)
        sorted_distances = ai_wknn_sort_distances(distances)
        neighbors = ai_wknn_neighbors(sorted_distances,k)
        class_weights = ai_class_weights(neighbors,y_train)
        predicted_class = ai_weighted_majority(neighbors,y_train,class_weights)
        predictions.append(predicted_class)
    return predictions


#Main Function
def main():
    data = pd.read_excel("thyroid_dataset.xlsx")

    X = data.drop(columns=["Condition","Record ID"])
    Y = data["Condition"]

    numerical = ["age","TSH","T3","TT4","T4U","FTI","TBG"]
    binary_columns = ["sex","on thyroxine","query on thyroxine","on antithyroid medication","sick","pregnant","thyroid surgery","I131 treatment",
                    "query hypothyroid","query hyperthyroid","lithium","goitre","tumor","hypopituitary","psych","TSH measured","T3 measured",
                    "TT4 measured","T4U measured","FTI measured","TBG measured"]
    nominal_columns = ["referral source"]
    categorical = (binary_columns+ nominal_columns)

    X_train, X_test, y_train, y_test = ai_data_split(X,Y)

    X_train, X_test = ai_wknn_missing_values(X_train,X_test,numerical,categorical)

    X_train = ai_wknn_label_encoding(X_train,binary_columns)
    X_test = ai_wknn_label_encoding(X_test,binary_columns)
    X_train, X_test = ai_wknn_one_hot_encoding(X_train,X_test,nominal_columns)

    X_train = X_train.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    k = 7 #int(input("Enter the value of k: "))

    predictions = ai_wknn_predict(X_train,y_train,X_test,k)

    print("AI Weighted k-NN Results")
    print("k =", k)
    prediction_counts = {}
    for prediction in predictions:
        if prediction not in prediction_counts:
            prediction_counts[prediction] = 0
        prediction_counts[prediction] += 1
    print("\nPredicted class counts:")
    for class_name, count in prediction_counts.items():
        print(class_name,":",count)

if __name__ == "__main__":
    main()