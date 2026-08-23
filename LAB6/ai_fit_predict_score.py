# AI Tool Used: ChatGPT
# Lab 6 - AI-generated fit, predict and score functions


from ai_knn import (ai_calculate_distances,ai_sort_distances,ai_knn_neighbors,ai_majority_knn)
from ai_wknn import (ai_wknn_calculate_distances,ai_wknn_sort_distances,ai_wknn_neighbors,ai_class_weights,ai_weighted_majority)

# AI Tool Used: ChatGPT
def ai_fit(X_train, y_train):
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of samples.")

    training_data = X_train
    training_labels = y_train
    return training_data, training_labels


# AI Tool Used: ChatGPT
def ai_predict(X_test, training_data, y_train, k):
    if len(training_data) != len(y_train):
        raise ValueError("Training data and y_train must contain the same number of samples.")

    if k <= 0 or k > len(training_data):
        raise ValueError("k must be between 1 and the number of training samples.")

    predictions = []
    for test_vector in X_test:
        distances = ai_calculate_distances(training_data,test_vector)
        sorted_distances = ai_sort_distances(distances)
        neighbors = ai_knn_neighbors(sorted_distances,k)
        predicted_class = ai_majority_knn(neighbors,y_train)
        predictions.append(predicted_class)
    return predictions


# AI Tool Used: ChatGPT
def ai_wknn_predict(X_test, training_data, y_train, k):
    if len(training_data) != len(y_train):
        raise ValueError("Training data and y_train must contain the same number of samples.")
    if k <= 0 or k > len(training_data):
        raise ValueError("k must be between 1 and the number of training samples.")

    predictions = []
    for test_vector in X_test:
        distances = ai_wknn_calculate_distances(training_data,test_vector)
        sorted_distances = ai_wknn_sort_distances(distances)
        neighbors = ai_wknn_neighbors(sorted_distances,k)
        class_weights = ai_class_weights(neighbors,y_train)
        predicted_class = ai_weighted_majority(neighbors,y_train,class_weights)
        predictions.append(predicted_class)
    return predictions


# AI Tool Used: ChatGPT
def ai_score(y_test, predictions):
    if len(y_test) != len(predictions):
        raise ValueError("y_test and predictions must contain the same number of samples.")
    if len(y_test) == 0:
        raise ValueError("y_test cannot be empty.")

    correct_predictions = 0
    for actual, predicted in zip(y_test,predictions):
        if actual == predicted:
            correct_predictions += 1
    accuracy = (correct_predictions / len(y_test))
    return accuracy


# MAIN PROGRAM
def main():
    # no data loading and preprocessing here
    print("AI fit, predict, weighted predict and score functions are ready.")


if __name__ == "__main__":
    main()