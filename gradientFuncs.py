import numpy as np
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

def norm_calc(output):
    delta = []
    for row in output:
        delta.append(np.linalg.norm(row))
    return np.array(delta)

def categorical_crossentropy(targets, outputs):

    samples = len(targets)
    classes = len(targets[0])
    
    for s in range(samples):
        for c in range(classes):
            if targets[s,c] == 1:
                outputs[s,c] -= 1

    
    return norm_calc(outputs)

def binary_crossentropy(targets, outputs):

    samples = len(targets)
    classes = len(targets[0])
    
    for s in range(samples):
        for c in range(classes):
            if targets[s,c] == 1:
                outputs[s,c] -= 1

    return norm_calc(outputs)

    
gradient_func_dict = { 'binary_crossentropy' : binary_crossentropy,
                   'categorical_crossentropy': categorical_crossentropy }
                #    'mean_squared_error' : mean_squared_error,
                #    'mean_absolute_error' : mean_absolute_error }

def get_loss_func(identifier): # function to return gradient function corresponding to 
    return gradient_func_dict[identifier]