import numpy as np
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

class BatchNormalization:
    def __init__(self, gamma, beta, epsilon = 1e-5, name = 'normalization'):
        self.gamma = gamma # gamma value of normalization
        self.beta = beta # beta value of normalization
        self.epsilon = epsilon # epsilon value of normalization
        self.layer_name = name
        

    def forward(self, X):
        sample_mean = X.mean(axis=0)
        sample_var = X.var(axis=0)
        
        X_norm = (X - sample_mean) / np.sqrt(sample_var + self.epsilon)
        self.output = self.gamma * X_norm + self.beta