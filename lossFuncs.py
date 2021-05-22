import numpy as np
import activationFuncs
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

# loss functions to be imported in class

def binary_crossentropy(target, output): # binary crossentropy error
    output = activationFuncs.sigmoid(output)
    return -np.multiply(target,np.log(output)) - np.multiply((1-target),np.log(1-output))


def categorical_crossentropy(target, output): # categorical cross entropy error
    # print('output:',output.shape)
    # output = activationFuncs.softmax(output)
    # print('softmax output: ', output.shape)
    # print('applied log: ', np.log(output).shape)
    # print('target:',target.shape)
    # print('multiplied:',np.multiply(target, np.log(output)).shape)
    # print('error:', -np.sum(np.multiply(target, np.log(output))).shape)
    return -np.sum(np.multiply(target, np.log(output)))


def mean_squared_error(target, output): # mean squared error
    return np.square(target-output)/target.shape[0]


def mean_absolute_error(target, output): # mean absolute error
    return np.abs(target-output)/target.shape[0]


loss_func_dict = { 'binary_crossentropy' : binary_crossentropy,
                   'categorical_crossentropy': categorical_crossentropy,
                   'mean_squared_error' : mean_squared_error,
                   'mean_absolute_error' : mean_absolute_error }


def get_loss_func(identifier): # function to return loss function corresponding to 
    return loss_func_dict[identifier]