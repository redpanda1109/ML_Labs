# AI Tool Used: ChatGPT
# Lab 6 - Performance comparison of three k-NN implementations - Accuracy, Precision, Recall, F-score, execution time
# The three implementations compared are: Lab5 self k-NN, Lab5 package k-NN, Lab6 AI-generated k-NN
# The comparison is performed for k = 7, 9 and 11.


import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from lab5_knn import (X_train,X_test,y_train,y_test,quick_sort,distance_train,k_neighbour,k_classes,winning_class)
from ai_knn import ai_knn_predict


# ACCURACY
# AI Tool Used: ChatGPT
def accuracy(y_actual, y_predicted):
    # Number of correct predictions / Total number of predictions 
    if len(y_actual) != len(y_predicted):
        raise ValueError("Actual and predicted labels must have the same length.")
    if len(y_actual) == 0:
        raise ValueError("The input labels cannot be empty.")

    correct = 0
    for actual, predicted in zip(y_actual,y_predicted):
        if actual == predicted:
            correct += 1
    return correct / len(y_actual)


# PRECISION
# AI Tool Used: ChatGPT
def precision(y_actual, y_predicted):
    # TP/ TP+FP
    if len(y_actual) != len(y_predicted):
        raise ValueError("Actual and predicted labels must have the same length.")

    true_positive = 0
    false_positive = 0
    for actual, predicted in zip(y_actual,y_predicted):
        if predicted == "CONDITION":
            if actual == "CONDITION":
                true_positive += 1
            else:
                false_positive += 1
    denominator = true_positive + false_positive
    if denominator == 0:
        return 0.0
    return true_positive / denominator


# RECALL
# AI Tool Used: ChatGPT
def recall(y_actual, y_predicted):
    # TP / TP+FN
    if len(y_actual) != len(y_predicted):
        raise ValueError("Actual and predicted labels must have the same length.")

    true_positive = 0
    false_negative = 0
    for actual, predicted in zip(y_actual,y_predicted):
        if actual == "CONDITION":
            if predicted == "CONDITION":
                true_positive += 1
            else:
                false_negative += 1
    denominator = true_positive + false_negative
    if denominator == 0:
        return 0.0
    return true_positive / denominator


# F-SCORE
# AI Tool Used: ChatGPT
def f_score(precision_value, recall_value):
    # 2*Precision*Recall / Precision+Result
    denominator = precision_value + recall_value
    if denominator == 0:
        return 0.0
    return (2 * precision_value * recall_value) / denominator

# EXECUTION TIME
# AI Tool Used: ChatGPT
def execution_time(prediction_function):
    # measure the execution time of the function
    start_time = time.perf_counter()
    predictions = prediction_function()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return predictions, elapsed_time



# LAB 5 SELF-WRITTEN K-NN PREDICTION
# AI Tool Used: ChatGPT
def lab5_knn_prediction(train_data, test_data, y_train, k):
    predictions = []
    for test_vector in test_data:
        distances = distance_train(train_data,test_vector)
        sorted_distances = quick_sort(distances)
        neighbours = k_neighbour(sorted_distances,k)
        class_type = k_classes(neighbours,y_train)
        prediction = winning_class(class_type)
        predictions.append(prediction)
    return predictions

def create_feature_vectors(data):
    feature_vectors = []
    for i in range(len(data)):
        vector = []
        for feature in data.columns:
            vector.append(data.iloc[i][feature])
        feature_vectors.append(vector)
    return feature_vectors


# PACKAGE K-NN PREDICTION
# AI Tool Used: ChatGPT
def package_knn_prediction(model, X_test):
    return model.predict(X_test)


# CALCULATE ALL METRICS
# AI Tool Used: ChatGPT
def calculate_metrics(y_actual, predictions):
    accuracy_value = accuracy(y_actual,predictions)
    precision_value = precision(y_actual,predictions)
    recall_value = recall(y_actual,predictions)
    f_score_value = f_score(precision_value,recall_value)
    return {
        "accuracy": accuracy_value,
        "precision": precision_value,
        "recall": recall_value,
        "f_score": f_score_value
    }


# AVERAGE VALUES
# AI Tool Used: ChatGPT
def calculate_average(values):
    if len(values) == 0:
        raise ValueError("Values cannot be empty.")
    return sum(values) / len(values)


# RUN ONE IMPLEMENTATION TEN TIMES
# AI Tool Used: ChatGPT
def run_experiment(implementation,k,train_data,test_data,y_train,y_train_values,y_test,test_data_for_package):
    accuracy_values = []
    precision_values = []
    recall_values = []
    f_score_values = []
    execution_times = []

    # Create package model once before timing.
    package_model = None
    if implementation == "Package k-NN":
        package_model = KNeighborsClassifier(n_neighbors=k)
        package_model.fit(train_data,y_train)

    # Perform 10 runs.
    for run in range(10):
        if implementation == "Lab 5 k-NN":
            prediction_function = lambda: lab5_knn_prediction(train_data,test_data,y_train,k)
        elif implementation == "Package k-NN":
            prediction_function = lambda: package_knn_prediction(package_model,test_data)
        elif implementation == "AI k-NN":
            prediction_function = lambda: ai_knn_predict(train_data,y_train_values,test_data,k)[0]
        else:
            raise ValueError("Unknown k-NN implementation.")

        predictions, elapsed_time = execution_time(prediction_function)
        metrics = calculate_metrics(y_test,predictions)

        accuracy_values.append(metrics["accuracy"])
        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
        f_score_values.append(metrics["f_score"])
        execution_times.append(elapsed_time)

    # Calculate averages.
    averages = {"accuracy": calculate_average(accuracy_values),
        "precision": calculate_average(precision_values),
        "recall": calculate_average(recall_values),
        "f_score": calculate_average(f_score_values),
        "execution_time": calculate_average(execution_times)
    }

    return {
        "accuracy": accuracy_values,
        "precision": precision_values,
        "recall": recall_values,
        "f_score": f_score_values,
        "execution_time": execution_times,
        "averages": averages
    }



# PRINT RESULTS
# AI Tool Used: ChatGPT
def print_results(implementation,results):
    print("\n" + implementation)
    print("-" * 70)
    print("Accuracy:")
    print(results["accuracy"])
    print("Average Accuracy:",results["averages"]["accuracy"])
    print("\nPrecision:")
    print(results["precision"])
    print("Average Precision:",results["averages"]["precision"])
    print("\nRecall:")
    print(results["recall"])
    print("Average Recall:",results["averages"]["recall"])
    print("\nF-score:")
    print(results["f_score"])
    print("Average F-score:",results["averages"]["f_score"])
    print("\nExecution Time:")
    print(results["execution_time"])
    print("Average Execution Time:",results["averages"]["execution_time"],"seconds")


# MAIN COMPARISON
def main():
    train_data = create_feature_vectors(X_train)
    test_data = create_feature_vectors(X_test)
    y_train_values = y_train.to_numpy()

    k_values = [7,9,11]
    implementations = ["Lab 5 k-NN","Package k-NN","AI k-NN"]

    # Run comparison for every k.
    for k in k_values:
        print("\n"+ "=" * 80)
        print("k =",k)
        print("=" * 80)

        for implementation in implementations:
            results = run_experiment(implementation,k,train_data,test_data,y_train,y_train_values,y_test,test_data)
            print_results(implementation,results)
        print("\n"+ "=" * 80)



# RUN PROGRAM
if __name__ == "__main__":
    main()