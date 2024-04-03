import numpy as np
from medpy import metric
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score

def calculate_dice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        return metric.binary.dc(pred, gt)
    elif pred.sum() == 0 and gt.sum()==0:
        return 1
    else:
        return 0


def calculate_hd95(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        return metric.binary.hd95(pred, gt)
    elif pred.sum() == 0 and gt.sum()==0:
        return 1
    else:
        return None
    
def single_object_evaluation(prediction, label):
    dice_test = f1_score(np.ndarray.flatten(np.array(label, dtype=bool)),
                          np.ndarray.flatten(prediction))
    miou_test = jaccard_score(np.ndarray.flatten(np.array(label, dtype=bool)),
                          np.ndarray.flatten(prediction))
    recall_test = recall_score(np.ndarray.flatten(np.array(label, dtype=bool)),
                           np.ndarray.flatten(prediction))
    accuracy_test = accuracy_score(np.ndarray.flatten(np.array(label, dtype=bool)),
                               np.ndarray.flatten(prediction))
    return dice_test, miou_test, recall_test, accuracy_test