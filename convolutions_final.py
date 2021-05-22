import numpy as np
import cv2 
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

# model.add(Conv2D(256,(3,3), input_shape=X.shape[1:]))

class Conv2D:
    def __init__(self, numFilters, filterDims, stride, paddingType, name, input_shape = None, prev_shape = 0):
        self.rowsFilter, self.colsFilter = filterDims # initialize filter dimensions
        self.numFilters = numFilters # initialize number of filters
        if input_shape == None: # to check if input shape is given, otherwise later compute from input data
            self.input_shape = input_shape    
        else:
            self.numImages, self.channelsFilter, self.rowsImage, self.colsImage = input_shape #for numpy
        # print(input_shape)
        self.stride = stride
        self.paddingType = paddingType
        self.layer_name = name
        


    def initialize_weights_biases(self):
        # initialize random float weights and biases between -1 and 1
        self.weights = np.random.uniform(-1,1, (self.numFilters, self.channelsFilter, self.rowsFilter, self.colsFilter)) # for numpy
        # self.weights = np.random.uniform(-1,1, (self.numFilters, self.rowsFilter, self.colsFilter, self.channelsFilter)) # for cv2
        self.biases = np.random.uniform(-1,1,self.numFilters)    
    

    def padImages(self, images):
        # if padding is valid, no need to pad
        if  self.paddingType == 'valid':
            self.padding = 0
        
        # else if padding is same, calculate the padding required on each side using output dim formula by keeping both input and output dims equal
        elif  self.paddingType == 'same':
            self.padding = int( images[0].shape[1]*(self.stride-1) + self.rowsFilter - 1 )/2
            images = np.pad(images, ((0,self.padding)), mode='constant')
            images = np.pad(images, ((self.padding,0)), mode='constant')
            images = images[self.padding:-self.padding]
        
        return images
            

    def compute_output_shapes(self):
        # compute the output shape of the convolutions
        # call after padding otherwise 'no self.padding attribute error'
        self.rowsConv = int((self.rowsImage + 2*self.padding - self.rowsFilter)//self.stride + 1)
        self.colsConv = int((self.colsImage + 2*self.padding - self.colsFilter)//self.stride + 1)    


    def convolution2d(self, image, filter, bias):
        convImage = np.zeros((self.rowsConv, self.colsConv))
        for y in range(self.rowsConv):
            if self.stride*y+self.rowsFilter > self.rowsImage:
                break # if filter is already past image, there is no point increasing y anymore
            for x in range(self.colsConv):
                if self.stride*x+self.colsFilter > self.colsImage:
                    break # if filter is already past image, there is no point increasing x anymore
                convImage[y,x] = np.sum(np.multiply(image[0:self.channelsFilter, self.stride*y:self.stride*y+self.rowsFilter, self.stride*x:self.stride*x+self.rowsFilter ], filter)) # for numpy array
                # convImage[y,x] = np.sum(np.multiply(image[self.stride*y:self.stride*y+self.rowsFilter, self.stride*x:self.stride*x+self.rowsFilter, 0:self.channelsFilter ], filter)) # for cv2 array

        return convImage+bias


    def forward(self, images, back_weights = None, back_biases = None):
        images = np.array(self.padImages(images)) # pad images if necessary and convert it to batch of images otherwise np notation will be used and first dim of cv notation will be discarded
        if hasattr(self, 'input_shape'):
            self.numImages, self.channelsFilter, self.rowsImage, self.colsImage = images.shape # when input data dimensions are not provided

        if np.any(back_weights) != None: # use back weights if available
            self.weights = back_weights
            self.biases = back_biases
        else: # only works first epoch
            self.initialize_weights_biases() # initialize random weights for given number of filters and image depth

        self.compute_output_shapes() # keep output shapes ready for convolution
        self.output = []
        for image in images:
            maps_for_image = []
            for weight, bias in zip(self.weights, self.biases): # apply each filter to every image
                maps_for_image.append(self.convolution2d(image, weight, bias))
            self.output.append(np.array(maps_for_image))
        self.output = np.array(self.output)
