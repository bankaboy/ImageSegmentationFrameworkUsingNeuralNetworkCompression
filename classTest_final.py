# script to import classes and train model

import pickle
import os
import sys




'''LOAD TEACHER MODEL'''
ROOT_DIR = os.path.abspath("Mask_RCNN/")

import warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.h5')

# Load weights trained on MS-COCO
model.load_weights('mask_rcnn_coco.h5', by_name=True)




'''LOAD TRAINING DATA'''
from data_preprocess import prepare_dataset

images_path = './data/training_images'
annotations_path = './data/annotations'
classes_file =  './data/classes.txt'
X, y = prepare_dataset(images_path, annotations_path, classes_file)




'''TRAINING PROCEDURE''' 
from models_final import Sequential
from convolutions_final import Conv2D
from normalizations import BatchNormalization
from poolings import MaxPool2D
from dense_final import Flatten, Dense
from activations import Activation

model = Sequential()
model.add(Conv2D(10,(3,3),1, "valid","convLayer1", X.shape))
model.add(MaxPool2D((2,2), 2, "valid", "poolLayer1"))
model.add(Activation('relu'))
model.add(BatchNormalization(1,0, 1e-5))

model.add(Conv2D(10,(3,3),1, "valid","convLayer2"))
model.add(MaxPool2D((2,2), 2, "valid", "poolLayer2"))
model.add(Activation('relu'))
model.add(BatchNormalization(1,0, 1e-5))

model.add(Conv2D(10,(3,3),1, "valid","convLayer3"))
model.add(MaxPool2D((2,2), 2, "valid", "poolLayer3"))
model.add(Activation('relu'))
model.add(BatchNormalization(1,0, 1e-5))

model.add(Conv2D(10,(3,3),1, "valid","convLayer4"))
model.add(MaxPool2D((2,2), 2, "valid", "poolLayer4"))
model.add(Activation('relu'))
model.add(BatchNormalization(1,0, 1e-5))

model.add(Flatten())
model.add(Dense(100,"denseLayer1"))
model.add(Activation('relu'))
model.add(BatchNormalization(1,0, 1e-5))

model.add(Dense(10, "denseLayer2"))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy')
model.fit(model, X, y, 4, 2)

# print('Model Output: ')
# pprint(model.output)
# print('Model Output: ', model.output)
# print('Output Layer Loss: ',model.lossLayer.output)

model.save()

with open('output.pickle', 'wb') as handle:
    pickle.dump(model.output, handle)
handle.close()

with open('targets.pickle', 'wb') as handle:
    pickle.dump(randTargets, handle)
handle.close()