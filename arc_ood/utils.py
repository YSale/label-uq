import numpy as np
import scipy
import sklearn.tree
import sklearn.ensemble as se
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import itertools
import config

import pandas as pd 

EPS = 1e-10


def accuracy(model, loader):
    correct = 0
    total = 0
    for inputs, targets in loader:
        outputs = model(inputs)
        outputs = outputs.mean(axis=2)
        correct += torch.sum(torch.argmax(outputs, dim=1) == targets)
        total += targets.shape[0]
    return correct/total


def torch_get_outputs(model, loader):
    #if config.DATA == "cifar":
    model.to("mps")
    for m in model.members:
        m.to("mps")
    outputs = torch.empty(0)
    targets = torch.empty(0)
    for input, target in loader:
        #if config.DATA == "cifar":
        input = input.to("mps")
        outputs = torch.cat((outputs, model(input).detach()), dim=0)
        targets = torch.cat((targets, target.detach()), dim=0)
    return outputs, targets



def append_array(old, new, axis):
    if old is None:
        return new
    else:
        return np.concatenate((old, new), axis=axis)









