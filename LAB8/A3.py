import numpy as np
import matplotlib.pyplot as plt
from A1 import summation_unit, activation_unit, comparator_unit

def calculate_error(errors):
    sse=0
    for i in errors:
        sse += i**2
    return sse

def train_perceptron(input, target, weights, learning_rate, epochs, activation_function):
    count=0
    epoch_errors=[]
    for epoch in range(1, epochs + 1):
        count+=1
        for i in range(len(input)):
            total = summation_unit(input[i], weights)
            output = activation_unit(total, activation_function)
            error = comparator_unit(target[i], output)
            
            weights[0] = weights[0] + learning_rate*error  #update the bias
            for j in range(1,len(weights)):
                weights[j] = weights[j] + learning_rate*error*input[i][j-1]  #update the weights
        
        errors=[]     #i get the errors or each input...put them in an array
        for i in range(len(input)):
            total = summation_unit(input[i], weights)
            output = activation_unit(total, activation_function)
            error = comparator_unit(target[i], output)
            errors.append(error)
        sse=calculate_error(errors)     #calculate the sse for that epoch using the input errors
        epoch_errors.append(sse)
        if sse <= 0.002:
            break
    return weights, count, sse, epoch_errors

def main():
    inputs=[[0,0],[0,1],[1,0],[1,1]]
    target1=[0,0,0,1]  #target for AND gate
    target2=[-1,-1,-1,1]  #target for bipolar AND gate
    activation_functions =["bipolar step", "sigmoid", "relu"]

    for i in range(len(activation_functions)):
        weights=[10,0.2, -0.75]  #weights for AND gate
        learning_rate = 0.05
        
        print("Training with activation function: ", activation_functions[i])
        if activation_functions[i] == "bipolar step":
            target = target2
        else:
            target = target1
        final_weights, epoch, final_sse, epoch_errors = train_perceptron(inputs, target, weights, learning_rate, 1000, activation_functions[i])
        print("Final Weights: ", final_weights)
        print("Total Epochs: ", epoch)
        print("Final SSE: ", final_sse)



if __name__ == "__main__":
    main()
