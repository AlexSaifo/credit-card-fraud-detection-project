import numpy as np
from layer import Layer
from utils import *

DELTA_MAX = 50
DELTA_MIN = 0

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.pre_derivative = np.zeros(output_size * input_size).reshape(output_size,input_size)
        self.pre_delta = np.full(self.pre_derivative.shape,0.0125)
        self.pre_bias_delta = np.full((output_size,1),0.0125)
        self.pre_bias_derivative = np.zeros(output_size).reshape(output_size,1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        db = np.mean( learning_rate * output_gradient,axis=1)
        db = db.reshape(db.shape[0],1)
        self.bias -= db
        return input_gradient

    def RProp(self, output_gradient, learning_rate):
        input_gradient = np.dot(self.weights.T, output_gradient)


        new_derivative = output_gradient.dot(self.input.T)
        self.pre_delta = calc_delta(np.sign(self.pre_derivative * new_derivative))
        self.pre_delta[self.pre_delta > DELTA_MAX] = DELTA_MAX
        self.pre_delta[self.pre_delta < DELTA_MIN] = DELTA_MIN
        dw = -np.sign(new_derivative) * self.pre_delta
        self.weights += 0.1*dw

        temp = np.mean(output_gradient,axis=1)
        new_derivative_b = temp.reshape(temp.shape[0],1)
        self.pre_bias_delta = calc_delta(np.sign(self.pre_bias_derivative * new_derivative_b))
        self.pre_bias_delta[self.pre_bias_delta > DELTA_MAX] = DELTA_MAX
        self.pre_bias_delta[self.pre_bias_delta < DELTA_MIN] = DELTA_MIN
        db = -np.sign(new_derivative_b) * self.pre_bias_delta
        self.bias += 0.1*db

        self.pre_derivative = new_derivative
        self.pre_bias_derivative = new_derivative_b
        return input_gradient