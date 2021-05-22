import numpy as np
import pickle
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

def quantize(weights_dict, biases_dict):
    for layer, weights in weights_dict.items():
        weights_dict[layer] = weights*128
        weights_dict[layer] = weights_dict[layer].astype('int8')

    for layer, biases in biases_dict.items():
        biases_dict[layer] = biases*128
        biases_dict[layer] = biases_dict[layer].astype('int8')

    return weights_dict, biases_dict