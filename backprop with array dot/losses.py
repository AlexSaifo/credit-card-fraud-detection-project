import numpy as np

def mse(y_true, y_pred, w0=1, w1=1):
    print ("mse: ",w0,w1)
    return np.mean( ((w1-w0)*y_true+w0) * np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred, w0=1, w1=1):
    print ("prime mse: ",w0,w1)
    return ((w1-w0)*y_true+w0) * 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
