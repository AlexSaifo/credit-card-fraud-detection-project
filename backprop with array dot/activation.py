import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # print("output gradient.shape: ",output_gradient.shape)
        # print("input.shape: ",self.input.shape)
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
    def RProp(self, output_gradient, learning_rate):
        # print("output gradient.shape Rpop: ",output_gradient.shape)
        return self.backward(output_gradient,learning_rate)