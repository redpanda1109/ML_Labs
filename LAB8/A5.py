import matplotlib.pyplot as plt
from A1 import summation_unit, activation_unit, comparator_unit
from A2 import train_perceptron

def main():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    target1 = [0, 1, 1, 0]  # target for XOR gate
    target2 = [-1, 1, 1, -1]  # target for bipolar XOR gate
    activation_functions = ["bipolar step", "sigmoid", "relu"]

    #A1
    print("A1:")
    a1_weights = [-1,1,1]  # weights for XOR gate
    for i in range(len(inputs)):
        total=summation_unit(inputs[i],a1_weights)
        output=activation_unit(total,"step")
        error=comparator_unit(target1[i],output)
        print("Input: ",inputs[i], "Target: ",target1[i], "Output: ",output," Error: ",error)

    #A2
    print("\nA2:")
    a2_weights = [10, 0.2, -0.75]  # weights for XOR gate
    learning_rate = 0.05
    final_weights, epoch, final_sse, epoch_errors = train_perceptron(inputs, target1, a2_weights, learning_rate, 1000, "step")
    print("Final Weights: ", final_weights)
    print("Total Epochs: ", epoch)
    print("Final SSE: ", final_sse)

    plt.plot(range(1, epoch + 1), epoch_errors)
    plt.xlabel('Epochs')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.show()

    #A3
    print("\nA3:")
    for i in range(len(activation_functions)):
        a3_weights=[10,0.2, -0.75]  #weights for XOR gate
        print("Training with activation function: ", activation_functions[i])
        if activation_functions[i] == "bipolar step":
            target = target2
        else:
            target = target1
        final_weights, epoch, final_sse, epoch_errors = train_perceptron(inputs, target, a3_weights, learning_rate, 1000, activation_functions[i])
        print("Final Weights: ", final_weights)
        print("Total Epochs: ", epoch)
        print("Final SSE: ", final_sse)


if __name__ == "__main__":
    main()

