import numpy as np
import math
from numba import jit

# model.add(MaxPooling2D(pool_size=(2,2)))

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

class Pooling2D:
    def __init__(self, poolDim, stride, paddingType, name = 'pooling', prev_shape = 0):
        self.rowsPool, self.colsPool = poolDim # size of pooling window
        self.stride = stride # stride of pooling window
        self.paddingType = paddingType # padding specified by user
        
        self.layer_name = name

    def padMaps(self, images):
        # if padding is valid, no need to pad
        if  self.paddingType == 'valid':
            self.padding = 0
        
        # else if padding is same, calculate the padding required on each side using output dim formula by keeping both input and output dims equal
        elif  self.paddingType == 'same':
            self.padding = int( images[0].shape[1]*(self.stride-1) + self.rowsPool - 1 )/2
            images = np.pad(images, ((0,self.padding)), mode='constant')
            images = np.pad(images, ((self.padding,0)), mode='constant')
            images = images[self.padding:-self.padding]
     
        return images        


    def applyPool(self, convMap):
        pass # later implemented in child classes


    def forward(self, convMaps):
        self.poolMaps = [] # array of output pool maps
        convMaps = np.array(self.padMaps(convMaps)) # apply padding to convMaps depending on input
        for convMap in convMaps: # apply pooling method to all convMaps in input
            self.poolMaps.append(self.applyPool(convMap))
        self.output = np.array(self.poolMaps)
        # for i in range(10):
        #     writeImage(f'{i}_{self.layer_name}.jpg', self.output[0][i])


class MaxPool2D(Pooling2D):
    def __init__(self, poolDim, stride, paddingType, name):
        super().__init__(poolDim, stride, paddingType, name)

    def applyPool(self, convMap):
        rowsOld, colsOld = convMap[0].shape # calculate the shape of feature map after pooling
        rowsNew = math.floor((rowsOld-self.rowsPool)/self.stride + 1)
        colsNew = math.floor((colsOld-self.colsPool)/self.stride + 1)

        poolMap = []
        for channel in convMap:
            layer = np.zeros((rowsNew, colsNew))
            for y in range(rowsNew):
                if self.stride*y+self.rowsPool > rowsOld:
                    break # if filter is already past image, there is no point increasing y anymore
                for x in range(colsNew):
                    if self.stride*x+self.colsPool > colsOld:
                        break # if filter is already past image, there is no point increasing x anymore
                    layer[y,x] = np.max(channel[self.stride*y:self.stride*y+self.rowsPool, self.stride*x:self.stride*x+self.colsPool])
            poolMap.append(layer)
        return np.array(poolMap)


class AveragePool2D(Pooling2D):
    def __init__(self, poolDim, stride, paddingType, name):
        super().__init__(poolDim, stride, paddingType, name)

    def applyPool(self, convMap):
        rowsOld, colsOld = convMap.shape
        rowsNew = math.floor((rowsOld-self.rowsPool)/self.stride + 1)
        colsNew = math.floor((colsOld-self.colsPool)/self.stride + 1)

        poolMap = []
        for channel in convMap:
            layer = np.zeros((rowsNew, colsNew))
            for y in range(rowsNew):
                if self.stride*y+self.rowsPool > rowsOld:
                    break # if filter is already past image, there is no point increasing y anymore
                for x in range(colsNew):
                    if self.stride*x+self.colsPool > colsOld:
                        break # if filter is already past image, there is no point increasing x anymore
                    layer[y,x] = np.average(channel[self.stride*y:self.stride*y+self.rowsPool, self.stride*x:self.stride*x+self.colsPool])
            poolMap.append(layer)
        return np.array(poolMap)