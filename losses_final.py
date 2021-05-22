import numpy as np
import lossFuncs
import gradientFuncs
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

class Loss:
    def __init__(self, loss, targets,soft_targets, outputs, name = 'loss', prev_shape = 0):
        # initialize loss function of class using predefined functions
        self.lossFunc = lossFuncs.get_loss_func(loss)
        self.gradientFunc = gradientFuncs.get_loss_func(loss)
        self.targets = np.nan_to_num(targets)
        self.outputs = np.nan_to_num(outputs)
        self.soft_targets = np.nan_to_num(soft_targets)
        self.layer_name = name


    def forward(self):
        # apply set loss to input
        generalLoss = self.lossFunc(self.targets, self.outputs)
        modelLoss = self.lossFunc(self.soft_targets, self.outputs)
        self.output = [generalLoss, modelLoss]
        self.gradients = self.gradientFunc(self.targets, self.outputs)