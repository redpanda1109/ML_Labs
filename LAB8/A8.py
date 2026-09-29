import numpy as np
from A1 import activation_unit

def train_network(inputs, target, weights, biases, learning_rate, max_epochs):
    # Initial weights
    v11, v12, v21, v22, w1, w2 = weights
    bh1, bh2, bo = biases

    for epoch in range(1, max_epochs + 1):
        errors = []
        for i in range(len(inputs)):
            A = inputs[i][0]
            B = inputs[i][1]

            # Forward propagation
            h1_total = A * v11 + B * v21 + bh1
            h2_total = A * v12 + B * v22 + bh2
            h1 = activation_unit(h1_total, "sigmoid")
            h2 = activation_unit(h2_total, "sigmoid")
            o_1 = h1 * w1 + h2 * w2 + bo
            output = activation_unit(o_1, "sigmoid")
            # Error
            error = target[i] - output
            errors.append(error)

            # Back propagation
            output_delta = error * output * (1 - output)
            h1_delta = h1 * (1 - h1) * output_delta * w1
            h2_delta = h2 * (1 - h2) * output_delta * w2
            # Update output weights
            w1 = w1 + learning_rate * output_delta * h1
            w2 = w2 + learning_rate * output_delta * h2
            bo = bo + learning_rate * output_delta
            # Update hidden weights
            v11 = v11 + learning_rate * h1_delta * A
            v21 = v21 + learning_rate * h1_delta * B
            v12 = v12 + learning_rate * h2_delta * A
            v22 = v22 + learning_rate * h2_delta * B
            # Update hidden biases
            bh1 = bh1 + learning_rate * h1_delta
            bh2 = bh2 + learning_rate * h2_delta

        # Calculate SSE after each epoch
        sse = 0
        for error in errors:
            sse = sse + error ** 2
        if sse <= 0.002:
            break
    print("Epoch:", epoch, "SSE:", sse)
    return v11, v12, v21, v22, w1, w2, bh1, bh2, bo, epoch, sse


def main():
    inputs = [[0, 0],[0, 1],[1, 0],[1, 1]]
    target = [0, 0, 0, 1]
    learning_rate = 0.05
    max_epochs = 1000
    weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    biases = [0.1, 0.1, 0.1]
    v11, v12, v21, v22, w1, w2, bh1, bh2, bo, epoch, sse = train_network(inputs, target, weights, biases, learning_rate,max_epochs)

    print("Final Weights:")
    print("V11:", v11)
    print("V12:", v12)
    print("V21:", v21)
    print("V22:", v22)
    print("W1:", w1)
    print("W2:", w2)

    print("\nFinal Biases:")
    print("BH1:", bh1)
    print("BH2:", bh2)
    print("BO:", bo)

    print("\nTotal Epochs:", epoch)
    print("Final SSE:", sse)

if __name__ == "__main__":
    main()