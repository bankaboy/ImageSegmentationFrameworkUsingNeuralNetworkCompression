import numpy as np
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

class Dense:
    def __init__(self, units, name = 'dense'):
        self.layer_name = name # initialize name of layer
        self.numUnits = units # initialize number of neurons in layer
        self.output = []
    

    def initialize_weights_biases(self, batch_size, prevUnits):
        # print('Inside dense initialize',batch_size, prevUnits)
        self.weights = np.random.uniform(-1,1,(batch_size, prevUnits, self.numUnits)) # initialize weights of dense layer
        self.biases = np.random.uniform(-1,1, (batch_size,self.numUnits)) # initialize biases of dense layer
        # print('Weights shape:',self.weights.shape)
        # print('Biases shape:',self.biases.shape)


    def forward(self, inputs, back_weights = None, back_biases = None):
        
        if np.any(back_weights) != None: # for first forward pass
            self.weights = back_weights
            self.biases = back_biases
        else: # only works in first epoch
            self.initialize_weights_biases(inputs.shape[0], inputs.shape[1]) # initialize random weights for given number of filters and image depth

        self.output = []

        # print('Len comparison:',self.weights.shape[0], self.biases.shape[0], inputs.shape[0])
        for weight, bias, vector in zip(self.weights, self.biases, inputs):
            self.output.append(np.dot(vector, weight)+bias)
        self.output = np.array(self.output)
        # print('Output Shape of Dense Layer: ', self.output.shape)


class Flatten:
    def __init__(self, name = 'flatten'):
        self.layer_name = name

        
    def forward(self, X):
        batch_size = X.shape[0]
        # print('Flatten Batch Size: ', batch_size)
        self.output = np.reshape(X, (batch_size,-1)) # flattens the input, does not return a layer