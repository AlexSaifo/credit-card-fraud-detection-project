from acc import acc_conf_matrix
import numpy as np
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, w0=1, w1=1, algorithm = "backward_propagation" ,verbose = True):
    for iteration in range(epochs):
        error = 0
        outputs = []
        x,y=x_train.T,y_train.T
        # forward
        output = predict(network, x)
        outputs.append(output)
        # error
        error = loss(y, output,w0,w1)

        # backward
        grad = loss_prime(y, output,w0,w1)
        if algorithm == "backward_propagation":
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
      
        elif algorithm == "RProp":
            iter = 1
            for layer in reversed(network):
                iter += 1
                grad = layer.RProp(grad, learning_rate)
       
        elif algorithm == "LPMProp":
            e= (np.array(error).T).reshape(error.shape[1])
            e= e.reshape(1,e.shape[0])
            iter = 1
            for layer in reversed(network):
                iter += 1
                grad = layer.LPMProp(grad, learning_rate, e)

        error = np.mean(error)
        error /= len(x_train)
        if verbose:
            print(f"{iteration + 1}/{epochs}, error={error}")
            acc_conf_matrix(y_train,outputs)



def test_my_model(network, loss, loss_prime, x_test, y_test, verbose = True):
    error = 0
    outputs = []
    x,y=x_test.T,y_test.T
    # forward
    output = predict(network, x)
    outputs.append(output)
    # error
    error = np.mean(loss(y, output))

    error /= len(x)
    if verbose:
        print(f" error={error}")
        acc_conf_matrix(y,outputs)


