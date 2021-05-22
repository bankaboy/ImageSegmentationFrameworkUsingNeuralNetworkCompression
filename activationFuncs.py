import numpy as np
from numba import jit

# activation functions to be imported in class

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def softmax(x): # softmax activation function
    exp_values = np.exp(x)
    output = exp_values/np.sum(exp_values,axis=1, keepdims=True)
    return output

def relu(x): # relu activation function
    return np.maximum(x,0)

def linear(x): # linear activation
    return x

def tanh(x): # tanh activation
    return np.tanh(x)

def sigmoid(x): # sigmoid activation
    output = 1/(1 + np.exp(-x))
    return output

activation_func_dict = { 'relu' : relu, 
                         'softmax' : softmax,
                         'linear' : linear,
                         'tanh' : tanh,
                         'sigmoid' : sigmoid }

def get_activation_func(identifier): # function to return activation function corresponding to identifier
    return activation_func_dict[identifier]