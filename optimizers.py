import numpy as np
from operator import itemgetter
from numba import jit


# get weights and preds from model and pass to layerloss
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

class Adam:
    '''
    Inputs to adam optimizer -
    student weights and biases to optimize
    gradients to use to go backwards (specific to the batch)
    batch number to select correct weights for optimization
    learning rate, momentum, weight decay
    teacher_weights matrix
    layer list
    '''

    def __init__(self, teacher_params, weights_dict, baises_dict, final_gradients, layer_list):
        self.teacher_params = teacher_params
        self.weights_dict = weights_dict
        self.biases_dict = baises_dict
        self.gradients =  final_gradients
        self.layer_list = layer_list
        

    def layer_loss_core(tl, sl):
        tl_norm = np.linalg.norm(tl)
        sl_norm = np.linalg.norm(sl)
        return np.sqrt(np.square(tl_norm)-np.square(sl_norm))


    def layer_loss(self, layer_name):
        tl_weights = self.teacher_params[layer_name][0]
        sl_weights = self.weights_dict[layer_name]
        tl_biases = self.teacher_params[layer_name][0]
        sl_biases = self.biases_dict[layer_name]
        weights_layer_error = self.layer_loss_core(tl_weights, sl_weights)
        biases_layer_error = self.layer_loss_core(tl_biases, sl_biases)

        return [weights_layer_error, biases_layer_error]


    def create_layer_loss_dict(self):
        self.layer_loss_dict = dict()
        for layer in self.layer_list:
            self.layer_loss_dict[layer] = self.layer_loss(layer)


    def pruning_params_helper(weights, allowed_prunes):
        indices = []
        norms = []
        i = 0
        for row in weights:
            indices.append(i)
            norms.append(np.linalg.norm(row)) # record the norm of each weight matrix and their index

        zipped = zip(indices, norms)
        zipped = list(zip)
        res = sorted(zipped, key = lambda x: x[1])
        indices = map(itemgetter(0),res)

        return indices[0:allowed_prunes]


    def pruning_params(self, batch_num):
        num_layers = len(self.layer_list)
        for num_layer in range(num_layers-2, 0, -1): # do not prune the output and input layer
            weights = self.weights_dict[self.layer_list[num_layer]][batch_num]

            for item_weights in weights:  # prune weights for each item in batch
                allowed_prunes = 5 # change according to need
                indices = self.pruning_params_helper(item_weights, allowed_prunes)
                
                # prune neurons at indices passed by helper (i.e delete entire row)
                del item_weights[indices,:]
                del self.weights_dict[self.layer_list[num_layer-1]][batch_num][:,indices]
                # also delete colums with same indices from previous dense matrix
                # rows - neurons from which input is coming
                # cols - neurons to which input is going
                # number of rows in a matrix is equal to number of cols in previous matrix
                # number of source unit in a matrix is equal to number of destination units in previous matrix

            self.weights_dict[self.layer_list[num_layer]][batch_num] = weights



    def backward(self, batch_num, gradients, learning_rate, momentum, decay, allow_pruning = False):
        for layer in reversed(self.layer_list):
            self.weights_dict[layer][batch_num] -= ( (learning_rate*momentum + decay)*gradients*self.weights_dict[layer][batch_num] + learning_rate*self.layer_loss_dict[layer][0])
            self.biases_dict[layer][batch_num] -= ( (learning_rate*momentum + decay)*gradients*self.biases_dict[layer][batch_num] + learning_rate*self.layer_loss_dict[layer][1])
        if allow_pruning:
            self.pruning_params(batch_num)
        else:
            self.output = [self.weights_dict, self.biases_dict]
