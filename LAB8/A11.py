from sklearn.neural_network import MLPClassifier

def main():
    inputs=[[0,0], [0,1], [1,0], [1,1]]
    target_and=[0, 0, 0, 1] #for AND gate
    target_xor=[0, 1, 1, 0] #for XOR gate
    and_mlp= MLPClassifier(hidden_layer_sizes=(2,),activation="logistic",solver="lbfgs",max_iter=1000,random_state=1)
    and_mlp.fit(inputs, target_and)
    and_predictions = and_mlp.predict(inputs)

    print("AND Gate:")
    for i in range(len(inputs)):
        print("Input:", inputs[i],"Target:", target_and[i],"Output:", and_predictions[i])
    print("AND Accuracy:", and_mlp.score(inputs, target_and))

    xor_mlp= MLPClassifier(hidden_layer_sizes=(2,),activation="logistic",solver="lbfgs",max_iter=1000,random_state=1)
    xor_mlp.fit(inputs, target_xor)
    xor_predictions = xor_mlp.predict(inputs)

    print("XOR Gate:")
    for i in range(len(inputs)):
        print("Input:", inputs[i],"Target:", target_xor[i],"Output:", xor_predictions[i])
    print("XOR Accuracy:", xor_mlp.score(inputs, target_xor))


if __name__ == "__main__":
    main()


