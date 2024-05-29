import numpy as np
import torch.nn as nn
import sklearn.metrics as sm
import uncertainty as unc
from typing import Union
import models as mds
import utils


def accuracy_rejection(model: Union[nn.Module, mds.RandomForest], x, y, unc_method):
    portion_vals = np.linspace(0, 1, 50, endpoint=False)

    acc_tu = np.empty(len(portion_vals))
    acc_eu = np.empty(len(portion_vals))
    acc_au = np.empty(len(portion_vals))
    acc_ra = np.empty(len(portion_vals))

    if isinstance(model, mds.RandomForest):
        preds = model.predict(x)
    elif isinstance(model, nn.Module):
        preds, y = utils.torch_get_outputs(model, x)
        preds = preds.detach().numpy()
        y = y.detach().numpy()
    if unc_method == "entropy":
        au = unc.aleatoric_uncertainty_entropy(preds)
        eu = unc.epistemic_uncertainty_entropy(preds)
        tu = unc.total_uncertainty_entropy(preds)
        # print("TU entropy", tu)
    elif unc_method == "variance":
        tu = unc.total_uncertainty_variance(preds)
        au = unc.aleatoric_uncertainty_variance(preds)
        eu = tu-au
    elif unc_method == 'lent':
        au = unc.uncertainty_lent(preds, loss = 'ent', kind = 'au')
        eu = unc.uncertainty_lent(preds, loss = 'ent', kind = 'eu')
        tu = unc.uncertainty_lent(preds, loss = 'ent', kind = 'tu')
        if not np.allclose(tu, (eu+au)):
            print('TU not EU + AU')
    
    preds = preds.mean(axis=2).argmax(axis=1)
    for i, portion in enumerate(portion_vals):
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, tu)
        acc_tu[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, au)
        #print(f'acc_preds: {acc_preds.shape}, acc_y: {acc_y.shape}')
        acc_au[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, eu)
        acc_eu[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_random(preds, y, portion)
        acc_ra[i] = sm.accuracy_score(acc_y, acc_preds)
    return np.asarray([acc_tu, acc_eu, acc_au, acc_ra])



def out_of_distribution(model, loader_id, loader_ood, unc_measure):
    preds_id, _ = utils.torch_get_outputs(model, loader_id)
    preds_ood, _ = utils.torch_get_outputs(model, loader_ood)
    if unc_measure == "entropy":
        uncertainties_id = unc.epistemic_uncertainty_entropy(preds_id)
        uncertainties_ood = unc.epistemic_uncertainty_entropy(preds_ood)
    elif unc_measure == 'variance':
        uncertainties_id = unc.epistemic_uncertainty_variance(preds_id)
        uncertainties_ood = unc.epistemic_uncertainty_variance(preds_ood)    
    elif unc_measure == 'lent':
        uncertainties_id = unc.uncertainty_lent(preds_id, loss = 'ent', kind = 'eu')
        uncertainties_ood = unc.uncertainty_lent(preds_ood, loss = 'ent', kind = 'eu')

    labels = np.concatenate((np.zeros(len(uncertainties_id)), np.ones(len(uncertainties_ood))))
    uncertainties = np.concatenate((uncertainties_id, uncertainties_ood))
    auroc = sm.roc_auc_score(labels, uncertainties)
    fpr, tpr, _ = sm.roc_curve(labels, uncertainties)
    return uncertainties_id, uncertainties_ood, auroc, fpr, tpr


