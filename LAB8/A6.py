import pandas as pd
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def perceptron_learning(inputs, target, weights, bias, epochs, learning_rate):
    for epoch in range(epochs):
        for i in range(len(inputs)):
            total = np.dot(inputs[i], weights) + bias
            output = sigmoid(total)
            error = target[i] - output

            # Update weights and bias
            weights += learning_rate * error * inputs[i]
            bias += learning_rate * error
    return weights, bias

def main():
    p=pd.read_csv('A6_customer_data.csv')
    inputs=p[["Candies", "Mangoes", "Milk Packets", "Payment"]].values
    target=p["High Value Tx?"].map({"Yes": 1, "No": 0}).values

    weights = np.array([0.1, 0.1, 0.1, 0.1])
    bias = 0.1
    learning_rate = 0.01
    epochs = 1000
    weights, bias = perceptron_learning(inputs, target, weights, bias, epochs, learning_rate)
    
    print("Final Weights:", weights)
    print("Final Bias:", bias)
    print("Predictions: ")
    for i in range(len(inputs)):
        total = np.dot(inputs[i], weights) + bias
        output = sigmoid(total)
        if output >= 0.5:
            prediction = "Yes"
        else:
            prediction = "No"
        print(f"Input: {inputs[i]}, Actual: {target[i]}, Prediction: {prediction}")


if __name__ == "__main__":
    main()