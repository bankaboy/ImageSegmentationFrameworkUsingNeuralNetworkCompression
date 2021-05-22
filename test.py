import pickle
import numpy as np
import argparse
import cv2
import os

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path', type=str, help='path of folder containing images to test on')
    parser.add_argument('-network_weights', type=str, help='name of weights file')
    parser.add_argument('-network_biases', type=str, help='name of weights file')
    parser.add_argument('-bbx_weights', type=str, help='name of bbx weights file')
    parser.add_argument('-bbx_biases', type=str, help='name of bbx weights file')
    parser.add_argument('-layer_classes', type=str, help='list of layer classes')
    parser.add_argument('-layer_names', type=str, help='list of layer names')
    parser.add_argument('-bbx_biases', type=str, help='name of bbx weights file')

args = get_arguments()

from convolutions_final import Conv2D
from dense_final import Dense, Flatten
from activations import Activation
from poolings import MaxPool2D
from normalizations import BatchNormalization
from nms import NonMaxSuppression
from bbox import BBXRegression
from models_final import Sequential

def load_file(path):
    with open(path, 'rb') as handle:
        contents = pickle.load(handle)
    handle.close()
    return contents


def assign_class(output):
        classes = []
        for row in output:
            classes.append( [np.maximum(row),np.argmax(row)] )

        return classes


weights_dict = load_file(args.network_weights)
biases_dict = load_file(args.network_biases)
bbx_weights = load_file(args.bbx_weights)
bbx_biases = load_file(args.bbx_biases)

model = Sequential()

for layer_name, layer_class in zip(args.layer_names, args.layer_classes):
    if hasattr('weights',layer_class):
        layer_class.weights = weights_dict[layer_name]
        layer_class.biases = biases_dict[layer_name]
    model.add(layer_class)


num_classes = weights_dict[args.layer_names[-1]].shape[1]
color_map = [tuple(np.random.randint(0,255,3)) for i in range(num_classes)]

def viewImage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize(image, predictions, filename):
    # preds are in format [class, prob, coords]
    for pred in predictions:
        classid = pred[0]
        pt1 = (pred[2][1], pred[2][0]) # x, y format (coords are in y,x format)
        pt2 = (pred[2][3], pred[2][2]) # x, y format (coords are in y,x format)
        cv2.rectangle(image,pt1, pt2, color_map[classid], thickness=-1) # fill the box
    viewImage(filename, image)

input = [[f,cv2.imread(f, cv2.IMREAD_GRAYSCALE)] for f in os.listdir(args.input_path)]


for filename, image in input:
    for layer in model.layers:
        org_image = image
        image = layer.forward(image)
        # non maximum suppresion : remove multiple detections for same object
        student_preds = assign_class(image)
        nms = NonMaxSuppression(student_preds, 0.3)
        nms.apply()
        final_student_preds = nms.output
        del nms
        
        # bounding box regression and modify bbox weights
        bbr = BBXRegression(bbx_weights, bbx_biases, None, final_student_preds)
        bbr.predict_box()
        final_preds = bbr.final_boxes
        del bbr

        visualize(org_image, final_preds, filename)