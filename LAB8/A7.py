import pandas as pd
import numpy as np
from A6 import sigmoid, perceptron_learning

def main():
    p=pd.read_csv('A6_customer_data.csv')
    inputs=p[["Candies", "Mangoes", "Milk Packets", "Payment"]].values
    target=p["High Value Tx?"].map({"Yes": 1, "No": 0}).values
    
    weights = np.array([0.1, 0.1, 0.1, 0.1])
    bias = 0.1
    learning_rate = 0.01
    epochs = 1000
    weights, bias = perceptron_learning(inputs, target, weights, bias, epochs, learning_rate)
    predictions=[]
    for i in range(len(inputs)):
        total = np.dot(inputs[i], weights) + bias
        output = sigmoid(total)
        if output >= 0.5:
            prediction = 1
        else:
            prediction = 0
        predictions.append(prediction)
        print(p["Customer"][i], "Actual:", target[i], "Prediction:", prediction)

    print("Matrix Pseudo Inverse")
    X = np.column_stack((np.ones(len(inputs)), inputs))
    pseudo_inverse = np.linalg.pinv(X)
    weights = np.dot(pseudo_inverse, target)
    for i in range(len(X)):
        output = np.dot(X[i], weights)
        if output >= 0.5:
            prediction = 1
        else:
            prediction = 0
        print(p["Customer"][i],"Actual:", target[i],"Prediction:", prediction)

if __name__ == "__main__":
    main()
    


