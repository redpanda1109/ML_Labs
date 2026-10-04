import numpy as np

def summation_unit(input,weights):
    sum=weights[0]
    for i in range(len(input)):
        sum=sum+input[i]*weights[i+1]
    return sum

def activation_unit(total, activation_function):
    if activation_function=="step":
        if total>=0:
            return 1
        else:
            return 0
    elif activation_function=="bipolar step":
        if total>0:
            return 1
        elif total==0:
            return 0
        else:
            return -1
    elif activation_function=="sigmoid":
        return 1/(1+np.exp(-total))
    elif activation_function=="tanh":
        return (np.exp(total)-np.exp(-total))/(np.exp(total)+np.exp(-total))
    elif activation_function=="relu":
        return max(0,total)
    elif activation_function=="leaky relu":
        if total>0:
            return total
        else:
            return 0.01*total   #0.01 is leaky factor
    else:
        print("Activation function not supported")

def comparator_unit(truth, output):
    error=truth-output
    return error


def main():
    inputs=[[0,0],[0,1],[1,0],[1,1]]
    weights=[-1, 1,1]  #weights for AND gate
    target=[0,0,0,1]  #target for AND gate
    for i in range(len(inputs)):
        total=summation_unit(inputs[i],weights)
        output=activation_unit(total,"step")
        error=comparator_unit(target[i],output)
        print("Input: ",inputs[i], "Target: ",target[i], "Output: ",output," Error: ",error)

if __name__ == "__main__":
    main()