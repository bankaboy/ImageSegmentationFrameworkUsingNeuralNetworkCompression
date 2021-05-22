from numba import jit
import numpy as np
import activationFuncs

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
class Activation:
    def __init__(self, activation, name = 'activation', prev_shape = 0):
        # initialize activation function of class using predefined functions
        self.activation = activationFuncs.get_activation_func(activation)
        self.layer_name = name
       
    def forward(self, x):
        # apply set activation to input
        self.output = self.activation(x)