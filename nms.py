import numpy as np
import yaml
import random
from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit

class NonMaxSuppression:
    def __init__(self, final_bbx, thresh):
        self.final_bbx = final_bbx
        self.threshold = thresh
        # final_bbx are in format [class_id, prob]


    def load_anchors(self):
        yfile = open('anchors.yaml')
        self.anchors_dict = yaml.load(yfile, Loader=yaml.FullLoader)


    # assign random anchor to detection (for now)
    def assign_anchors(self):
        for i in range(len(self.student_predicts)):
            anchor = random.choice(list(self.anchors_dict.values()))
            self.student_predicts[i].append(anchor)


    def IOU(self, box1, box2):
        xi1 = np.maximum(box1[0],box2[0])
        yi1 = np.maximum(box1[1],box2[1])
        xi2 = np.minimum(box1[2],box2[2])
        yi2 = np.minimum(box1[3],box2[3])
        inter_area = (yi2-yi1)*(xi2-xi1)

        box1_area = (box1[3]-box1[1])*(box1[2]-box1[0])
        box2_area = (box2[3]-box2[1])*(box2[2]-box2[0])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area/union_area
    
        return iou


    def apply(self):

        self.assign_anchors()

        def sortFunc(e):
            return e[1] # sort the list of bboxes accroding to probability
        self.final_bbx.sort(key = sortFunc)

        indices = [] 
        for i in range(len(self.final_bbx)-1, 0, -1):   # start from bottom (detections with lowest probability)
            item1 = self.final_bbx[i]
            for j in range(len(self.final_bbx)):
                item2 = self.final_bbx[j]
                if self.IOU(item1[2], item2[2]) >= self.threshold:
                    indices.append(i)

        self.final_bbx = np.array(self.final_bbx)
        del self.final_bbx[indices]

        self.output = self.final_bbx

            