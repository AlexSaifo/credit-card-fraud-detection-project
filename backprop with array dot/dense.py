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
        self.mu = 1
        self.pre_error = 0

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias


    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        # print (f"come grad = {output_gradient.shape},  grad= {weights_gradient.shape} , weights= {self.weights.shape}")
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
    
    
    
    def LPMProp(self, output_gradient, learning_rate, error):
        '''
            output_gradient shape = number of neuron in this layer * number of sample
            error shape = 1 * number of sample
            self.weights shape = number of neuron in this layer (output) * number of input
            self.bias shape = number of neuron in this layer (output) * 1
            self.input shape = number of input * number of sample
            
        '''
        jac  = np.zeros((error.shape[1],self.weights.shape[0]*self.weights.shape[1]))
        for p in range(error.shape[1]):
            for k in range(self.weights.shape[0]*self.weights.shape[1]):
                i = k//self.weights.shape[1]
                j = k%self.weights.shape[1]
                jac[p][k] =  output_gradient[i][p]* self.input[j][p]
                
        
        
        dw =  np.linalg.inv(np.dot(jac.T,jac) + (self.mu * np.eye(jac.shape[1])))
        dw = np.dot(dw,jac.T)
        dw = np.dot(dw,error.T)
        dw = dw.reshape(self.weights.shape[0],self.weights.shape[1])
        self.weights -= dw
       
       
       
       
       
       
        jac_b  = np.zeros((error.shape[1],self.weights.shape[0]))
        for p in range(error.shape[1]):
            for k in range(self.weights.shape[0]):
                jac_b[p][k] = output_gradient[k][p] * 1
                
        
        
        db =  np.linalg.inv(np.dot(jac_b.T,jac_b) + (self.mu * np.eye(jac_b.shape[1])))
        db = np.dot(db,jac_b.T)
        db = np.dot(db,error.T)
        self.bias -= db
        
        
       
        # error = np.sum(error)
        # if(error<=self.pre_error):
        #     self.mu /=10
        # else:
        #     self.mu*=10            
        # self.pre_error=error

        
        
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient
