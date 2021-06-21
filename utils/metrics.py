import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def micro_f1(output, labels, index):

    label, count = np.unique(labels, return_counts=True)    
    most_freq = np.argmax(count)    
    index = [i for i in index if labels[i] != most_freq]

    preds = output.max(1)[1]

    return f1_score(labels[index], preds[index], average='micro')

