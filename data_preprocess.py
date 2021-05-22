# filename,width,height,class,xmin,ymin,xmax,ymax
import os
import cv2
import numpy as np


def return_onehot(class_list, class_name):
    index = class_list.index(class_name)
    onehot = np.zeros(len(class_list))
    onehot[index] = 1
    return onehot


def prepare_dataset(images_path, annotations_filepath, class_file):
    
    classes = open(class_file, 'r').readlines() # needed for generating hard targets
    images_dict = dict()
    for image_filename in os.listdir(images_path): # later used to get crop from mentioned image in annotation
        images_dict[image_filename.split('/')[-1]] = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    annotations = open(annotations_filepath, 'r').readlines()


    # steps to prepare X and y
    X, y = [], []
    for annotaion in annotations:
        filename, w, h, class_name, xmin, ymin, xmax, ymax = list(annotaion.split(','))
        X.append(images_dict[filename][xmin:xmax, ymin:ymax])
        y.append(return_onehot(classes, class_name))

    return X, y
        




    