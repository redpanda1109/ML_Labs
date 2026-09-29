from A1 import activation_unit
from A2 import calculate_error

def train_network(inputs, target, weights, biases, learning_rate, max_epochs):
    v11, v12, v21, v22 = weights[:4]
    w11, w12, w21, w22 = weights[4:]
    bh1, bh2, bo1, bo2 = biases

    for epoch in range(1, max_epochs + 1):
        errors=[]
        for i in range(len(inputs)):
            A=inputs[i][0]
            B=inputs[i][1]

            #forward propagation
            h1_total = A * v11 + B * v21 + bh1
            h2_total = A * v12 + B * v22 + bh2
            h1 = activation_unit(h1_total, "sigmoid")
            h2 = activation_unit(h2_total, "sigmoid")
            o1_total = h1 * w11 + h2 * w21 + bo1
            o2_total = h1 * w12 + h2 * w22 + bo2
            o1 = activation_unit(o1_total, "sigmoid")
            o2 = activation_unit(o2_total, "sigmoid")

            error1 = target[i][0] #for o1......for error calculation
            error2 = target[i][1] #for o2
            errors.append(error1)
            errors.append(error2)

            #back propagation
            delta1 = error1 * o1 * (1 - o1)
            delta2 = error2 * o2 * (1 - o2)
            h1_delta = h1 * (1 - h1) * (w11 * delta1 + w12 * delta2)
            h2_delta = h2 * (1 - h2) * (w21 * delta1 + w22 * delta2)
            w11 = w11 + learning_rate * delta1 * h1
            w12 = w12 + learning_rate * delta2 * h1
            w21 = w21 + learning_rate * delta1 * h2
            w22 = w22 + learning_rate * delta2 * h2
            bo1 = bo1 + learning_rate * delta1
            bo2 = bo2 + learning_rate * delta2
            bh1 = bh1 + learning_rate * h1_delta
            bh2 = bh2 + learning_rate * h2_delta

            v11 = v11 + learning_rate * h1_delta * A
            v21 = v21 + learning_rate * h1_delta * B
            v12 = v12 + learning_rate * h2_delta * A
            v22 = v22 + learning_rate * h2_delta * B

        sse = calculate_error(errors)
        if sse <= 0.002:
            break
    return v11, v12, v21, v22, w11, w12, w21, w22, bh1, bh2, bo1, bo2, epoch, sse

def main():
    inputs=[[0,0], [0,1], [1,0], [1,1]]
    weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    biases=[0.1, 0.1, 0.1, 0.1]
    target=[[1,0], [1, 0], [1, 0], [0, 1]]
    learning_rate = 0.05
    max_epochs = 1000

    v11, v12, v21, v22, w11, w12, w21, w22, bh1, bh2, bo1, bo2, epoch, sse = train_network(inputs, target, weights, biases, learning_rate, max_epochs)

    print("Final Weights:")
    print("V11:", v11)
    print("V12:", v12)
    print("V21:", v21)
    print("V22:", v22)
    print("W11:", w11)
    print("W12:", w12)
    print("W21:", w21)
    print("W22:", w22)

    print("\nFinal Biases:")
    print("BH1:", bh1)
    print("BH2:", bh2)
    print("BO1:", bo1)
    print("BO2:", bo2)

    print("\nTotal Epochs:", epoch)
    print("Final SSE:", sse)


if __name__ == "__main__":
    main()

        