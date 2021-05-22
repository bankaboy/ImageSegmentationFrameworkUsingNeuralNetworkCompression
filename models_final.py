import numpy as np 
from pprint import pprint
from optimizers import Adam
from losses_final import Loss
import pickle
from bbox import BBXRegression
from nms import NonMaxSuppression
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

class Sequential:
    def __init__(self):
        self.weights_dict = dict() # create dictionary to store layer wise weights for easy backpropagation
        self.biases_dict = dict() # create dictionary to store layer wise biases for easy backpropagation
        self.layers = [] # save list of layers for sequential execution in fit method
        self.layer_names = [] # save list of layer names for backpropagation and setting weights
        self.layer_outputs = dict() # used during back propagation
        self.teacher_weights = [] # list of teacher weights to prevent running repeatedly, indexed by batch_num
        self.soft_targets = [] # list of teacher outputs weights to prevent running repeatedly, indexed by batch_num
        self.teacher_bboxes = [] # list of teacher bounding boxes to prevent running repeatedly, indexed by batch_num


    def getLayers(self): # function to show dtack of layers in model
        pprint(self.layer_names) 


    def add(self, layer): # add layer to network
        self.layers.append(layer)    # add layer to layer list
        self.layer_names.append(layer.layer_name)   # add layer name to layer name list
        self.weights_dict[layer.layer_name] = [] # make empty list for weights of all layers in network
        self.biases_dict[layer.layer_name] = [] # make empty list for biases of all layers in network


    def compile(self, loss):
        self.lossFunc = loss # loss function to use for model


    def return_preds_and_weights(self, model, proposals):
        # Run detection using teacher model
        results = model.detect([proposals], verbose=1)
        r = results[0]
        soft_targets = r['scores']
        bboxes = r['rois']
        layer_weights = model.get_trainable_layers()

        return soft_targets, layer_weights, bboxes


    # remove the empty entries of layers such as pooling
    def remove_nulls(self): 
        remove_keys = []
        for k in self.weights_dict.keys():
            if len(self.weights_dict[k]) == 0:
                remove_keys.append(k)

        for k in remove_keys:
            del self.weights_dict[k]
            del self.biases_dict[k] 


    # assign class to each output and their probability
    def assign_class(output):
        classes = []
        for row in output:
            classes.append( [np.maximum(row),np.argmax(row)] )

        return classes


    # create the weights for bounding box regression
    def create_bbx_weights(self, num_classes):
        self.bbx_weights = np.random.uniform(-1,1,(num_classes, 4))
        self.bbx_biases = np.random.uniform(-1,1,num_classes)


    # def fit(self, X, y, batch_size, epochs, learning_rate = 0.01, decay = 0.1, momentum = 0.9, validtion_split = 0):
    def fit(self, teacher_model, X, y, batch_size, epochs, learning_rate = 0.01, decay = 0.0002, momentum = 0.9, validtion_split = 0):
         
        self.create_bbx_weights(y.shape[1]) # y.shape[1] will give number of classes

        # first epoch done separately to create weights and bias matrix first
        epochs_loss = 0
        print('\n Epoch 0 \n')
        for i in range(int(len(X)/batch_size)):
            print('Batch',i)
            output = X[i*batch_size: (i+1)*batch_size] # internal buffer to keep track of input

            for layer in self.layers:
                layer.forward(output)
                output = layer.output
                # self.layer_outputs[layer.layer_name] = output # if you want to preserve output
                # print(layer.layer_name, output.shape)

                if hasattr(layer, 'weights'): # if the layer contains learnable parameters
                    self.weights_dict[layer.layer_name].append(layer.weights) # add the weights to the list
                    self.biases_dict[layer.layer_name].append(layer.biases)
                    self.layer_names.append(layer.layer_name) # add the name to the list

            self.output = output

            # soft targets for model loss
            # teacher layer weights for layer losses
            # bboxes for bbx regression
            teacher_params, soft_targets, teacher_bboxes = self.return_preds_and_weights(teacher_model, X[i*batch_size: (i+1)*batch_size])
            self.teacher_weights.append(teacher_params)
            self.soft_targets.append(soft_targets)
            self.teacher_bboxes.append(teacher_bboxes)

            # perform loss calcs
            self.lossLayer = Loss(self.lossFunc, y[i*batch_size: (i+1)*batch_size], soft_targets, self.output)
            self.lossLayer.forward()
            self.generalLoss, self.modelLoss = self.lossLayer.output
            final_gradients = self.lossLayer.gradients     # needed to start back propagation
            self.final_loss = 0.35*self.generalLoss + 0.65*self.modelLoss   # give more importance to model loss (soft targets)
            epochs_loss += self.final_loss

            self.remove_nulls() # remove layer entries with no params
            
            # back-propagation
            optimizer = Adam(teacher_params, 0, self.weights_dict, self.biases_dict, final_gradients, self.layer_names)
            optimizer.backward(i, final_gradients,learning_rate, decay, momentum)
            self.weights_dict, self.biases_dict = optimizer.outputs
            del optimizer


            # non maximum suppresion : remove multiple detections for same object
            student_preds = self.assign_class(output)
            nms = NonMaxSuppression(student_preds, 0.3)
            nms.apply()
            final_student_preds = nms.output
            del nms


            # bounding box regression and modify bbox weights
            bbr = BBXRegression(self.bbx_weights, self.bbx_biases, teacher_bboxes, final_student_preds)
            bbr.forward(learning_rate, momentum, decay)
            self.bbx_weights, self.bbx_biases = bbr.output
            del bbr


        print('Loss of epoch 1:', epochs_loss)
        print()

        teacher_model.close() # no longer required


        prev_epoch_loss = epochs_loss # compare loss of epoch with previous one
        threshold = 0.01 # threshold of difference in loss for two epochs
        plateau_iterations = 0 # keep track of number of iterations where the loss plateaus

        for epoch in range(epochs-1):
            print('\n Epoch', epoch+2, '\n')
            epochs_loss = 0 # to display loss after each epoch
            
            for i in range(int(len(X)/batch_size)):   # for the first batch (to create the weight and bias matrices of entire network)
                print('Batch',i)
                output = self.input[i*batch_size: (i+1)*batch_size] # internal buffer to keep track of input
                
                for layer in self.layers:
                    if hasattr(layer, 'weights'): # if the layer uses weights, then use the existing ones
                        back_weights = np.array(self.weights_dict[layer.layer_name][i])
                        back_biases = np.array(self.biases_dict[layer.layer_name][i])
                        # print(back_weights.shape)
                        # print(back_biases.shape)
                        layer.forward(output, back_weights, back_biases)
                    else:
                        layer.forward(output)

                    output = layer.output
                    print('Output shape:', output.shape)

                self.output = output


                # perform loss calcs
                self.lossLayer = Loss(self.lossFunc, y[i*batch_size: (i+1)*batch_size], self.soft_targets[i], self.output)
                self.lossLayer.forward()
                self.generalLoss, self.modelLoss = self.lossLayer.output
                final_gradients = self.lossLayer.gradients     # needed to start back propagation
                self.final_loss = 0.35*self.generalLoss + 0.65*self.modelLoss   # give more importance to model loss (soft targets)
                epochs_loss += self.final_loss
                
                self.remove_nulls() # remove layer entries with no params

                if epoch/epochs > 0.85: # allow pruning in the last 15% of training
                    allow_pruning = True


                # back-propagation : work on backpropagation of neuron weights
                optimizer = Adam(self.teacher_weights[i], self.weights_dict, self.biases_dict, final_gradients, self.layer_names)
                optimizer.backward(i, final_gradients, learning_rate, decay, momentum, allow_pruning)
                self.weights_dict, self.biases_dict = optimizer.output
                del optimizer

                
                # non maximum suppresion : remove multiple detections for same object
                student_preds = self.assign_class(output)
                nms = NonMaxSuppression(student_preds, 0.3)
                nms.apply()
                final_student_preds = nms.output
                del nms


                # bounding box regression and modify bbox weights
                bbr = BBXRegression(self.bbx_weights, self.bbx_biases, self.teacher_bboxes, final_student_preds)
                bbr.forward(learning_rate, momentum, decay)
                self.bbx_weights, self.bbx_biases = bbr.output
                del bbr


            if abs(prev_epoch_loss-epochs_loss) <= threshold:
                plateau_iterations += 1 # if the difference in loss is small increase number of plateau epochs
            if threshold == 15: # if 15 epochs had almost same loss
                learning_rate/=10 # divide learning rate by 10
            prev_epoch_loss = epochs_loss

            print('Loss of epoch',epoch+2,':', epochs_loss)
            print()

        


    def save(self):
        with open('weights.pickle', 'wb') as handle:
            pickle.dump(self.weights_dict, handle)
        handle.close()

        with open('bbx_weights.pickle', 'wb') as handle:
            pickle.dump(self.bbx_weights, handle)
        handle.close()
        
        with open('biases.pickle', 'wb') as handle:
            pickle.dump(self.biases_dict, handle)
        handle.close()

        with open('bbx_biases.pickle', 'wb') as handle:
            pickle.dump(self.bbx_biases, handle)
        handle.close()
        
        with open('loss.pickle', 'wb') as handle:
            pickle.dump(self.final_loss, handle)
        handle.close()

        with open('layer_names.pickle', 'wb') as handle:
            pickle.dump(self.layer_names, handle)
        handle.close()

        with open('layer_classes.pickle', 'wb') as handle:
            pickle.dump(self.layers, handle)
        handle.close()

        with open('layer_outputs.pickle', 'wb') as handle:
            pickle.dump(self.layer_outputs, handle)
        handle.close()