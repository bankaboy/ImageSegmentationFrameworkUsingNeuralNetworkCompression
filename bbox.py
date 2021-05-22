import numpy as np
import random
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

# student_predicts are list of lists of class detected and probability [class, prob, list of coordinates]
# teacher bbx is list of [class, (tuple of coordinates)]

class BBXRegression:
    def __init__(self, bbx_weights, bbx_biases, teacher_bbx, student_predicts):
        self.bbx_weights = bbx_weights
        self.bbx_biases = bbx_biases
        self.teacher_bbx = teacher_bbx
        self.student_predicts = student_predicts
        

    def prepare_data(self):
        for k,v in self.teacher_bbx:
            self.teacher_bbx[k] = list(v)
    
        def sortFunc(e):
            return e[0] # sort the list of bboxes accroding to classes
        self.teacher_bbx.sort(key = sortFunc)
        self.student_predicts.sort(key = sortFunc)


    def forward(self,learning_rate, momentum, decay): # used during training
        self.prepare_data()
        for student, teacher in zip(self.student_predicts, self.teacher_bbx):
            gradients = np.sqrt( np.square(np.array(teacher[2])) - np.square(np.array(student[2])) )
            class_id = student[0]
            self.bbx_weights[class_id] -=  (learning_rate*momentum + decay)*gradients*self.bbx_weights[class_id]
            self.bbx_biases[class_id] -=  (learning_rate*momentum + decay)*gradients*self.bbx_biases[class_id]

        self.output = self.bbx_weights, self.bbx_biases


    def predict(self): # used during testing
        self.final_boxes = []
        for pred in self.student_predicts:
            coords = np.array(pred[2]) + np.array(pred[2])*self.bbx_weights[pred[0]] + self.bbx_biases[pred[0]]
            self.final_boxes.append([pred[0], pred[1], coords])