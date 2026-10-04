import matplotlib.pyplot as plt
from A2 import train_perceptron

def main():
    inputs=[[0,0],[0,1],[1,0],[1,1]]
    target=[0,0,0,1]  #target for AND gate
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    for learning_rate in learning_rates:
        weights = [10, 0.2, -0.75]
        # implementing A2 where perceptron learning is happening...just changing the learning rates
        final_weights, epoch, final_sse, epoch_errors = train_perceptron( inputs,target,weights, learning_rate,1000, "step")
        results.append(epoch)
        print("Learning Rate:", learning_rate, ", Total Epochs:", epoch)

    plt.plot(learning_rates, results, marker="o")
    plt.xlabel('Learning Rate')
    plt.ylabel('Number of Epochs to Converge')
    plt.xticks(learning_rates, [str(rate) for rate in learning_rates])      #to het every learning rate point in the x-axis to be shown
    plt.show()

if __name__ == "__main__":
    main()