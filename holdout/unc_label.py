import numpy as np
import torch
from scipy.stats import entropy
from tqdm import tqdm
import math
import models as mds
import itertools


# Label-wise variance-based measures
def total_uncertainty_variance(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    mean = np.mean(probs, axis=2)
    t_u = mean * (1-mean)
    return t_u 
    

def aleatoric_uncertainty_variance(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    mean = np.mean(probs, axis=2)
    a_u = np.mean(probs*(1-probs), axis=2)
    #a_u = a_u / (mean)
    #a_u = a_u / (1-mean)
    return a_u 

def epistemic_uncertainty_variance(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    # S = np.sum(probs, axis = 1, keepdims = True) 
    # m = probs/S
    # e_u = np.sum(m*(1-m), axis=1) / (S.squeeze()+1)
    mean = np.mean(probs, axis=2, keepdims=1)
    e_u = np.mean(probs*(probs-mean), axis=2)
    #e_u = e_u / (mean).squeeze()
    #e_u = e_u / (1-mean).squeeze()
    return e_u 




