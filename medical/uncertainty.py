import os
import pickle
import random
import shutil

import scipy.ndimage
from PIL import Image
from scipy.stats import entropy

import models
import data_loading as dl
import plotter

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms as T


# variable probs with shape [batch_size, num_classes, num_ensemble_members]
def total_uncertainty_entropy(probs):
    t_u = entropy(np.mean(probs, axis=2), axis=1, base=2) / np.log2(probs.shape[1])
    return t_u


def epistemic_uncertainty_entropy(probs):
    mean_probs = np.mean(probs, axis=2)
    mean_probs = np.repeat(np.expand_dims(mean_probs, 2), repeats=probs.shape[2], axis=2)
    e_u = entropy(probs, mean_probs, axis=1, base=2) / np.log2(probs.shape[1])
    e_u = np.mean(e_u, axis=1)
    return e_u


def aleatoric_uncertainty_entropy(probs):
    a_u = entropy(probs, axis=1, base=2) / np.log2(probs.shape[1])
    a_u = np.mean(a_u, axis=1)
    return a_u


def uncertainty_1vsall_entropy(probs, kind='tu'):
    l_fun = lambda y: 0 if y == 0 else -np.log2(y)
    l_fun = np.vectorize(l_fun)
    k = probs.shape[1]
    fact = np.log2(k) + (k - 1) * np.log2(k / (k - 1))

    # probs shape (N,K,M)
    ps = np.stack([probs, 1 - probs])
    # stack shape (2,N,K,M)

    if kind == 'tu':
        out = np.sum(ps.mean(-1) * l_fun(ps.mean(-1)), 2).sum(0)
    elif kind == 'au':
        out = np.sum(ps * l_fun(ps), 2).mean(-1).sum(0)
    elif kind == 'eu':
        out = np.sum((ps * (l_fun(ps.mean(-1, keepdims=True)) - l_fun(ps))), 2).mean(-1).sum(0)
    return out / fact


def total_uncertainty_variance(probs):
    mean = np.mean(probs, axis=2)
    t_u = np.sum(mean * (1 - mean), axis=1)
    return t_u


def aleatoric_uncertainty_variance(probs):
    a_u = np.mean(np.sum(probs * (1 - probs), axis=1), axis=1)
    return a_u


def epistemic_uncertainty_variance(probs):
    mean = np.mean(probs, axis=2, keepdims=True)
    e_u = np.mean(np.sum(probs * (probs - mean), axis=1), axis=1)
    return e_u


def total_uncertainty_variance_label(probs):
    mean = np.mean(probs, axis=2)
    t_u = mean * (1 - mean)
    return t_u


def aleatoric_uncertainty_variance_label(probs):
    a_u = np.mean(probs * (1 - probs), axis=2)
    return a_u


def epistemic_uncertainty_variance_label(probs):
    mean = np.mean(probs, axis=2, keepdims=True)
    e_u = np.mean(probs * (probs - mean), axis=2)
    return e_u


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_PATH = '/models/'
RESULTS_PATH = '/results/'
AUTOPET_NORMALIZE = T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
INV_NORMALIZE = T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

supervise_train_transform = T.Compose([
    T.Resize(232, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(60),
    T.ToTensor(),
    AUTOPET_NORMALIZE
])

supervise_test_transform = T.Compose([
    T.Resize(232, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    AUTOPET_NORMALIZE
])
# get dataloader
dataloader = dl.get_dataloaders(data_path='/data/', batch_size=50,
                                train_transform=supervise_train_transform,
                                test_transform=supervise_test_transform)['test']
dataset_size = len(dataloader.dataset)
class_names = dataloader.dataset.classes
class_labels = ['Lung Cancer', 'Lymphoma', 'Melanoma', 'Negative']
num_classes = len(class_names)
file_names = dataloader.dataset.imgs
print(f'Number samples: {len(file_names)}')

# load model_names/paths
model, _, _, _ = models.get_resnet50(num_classes=num_classes)

num_ensembles = 5
model_list = sorted(os.listdir(MODELS_PATH))[:25]
models_per_ensemble = len(model_list) // num_ensembles
print('Following models used for computing uncertainty: ')
print(model_list)

# compute probabilities
y_pred_probs_mult_ens = np.zeros((dataset_size, num_classes, models_per_ensemble, num_ensembles))
for ensemble_index in range(num_ensembles):
    ens_models = model_list[ensemble_index * models_per_ensemble:(ensemble_index + 1) * models_per_ensemble]
    for model_idx in range(len(ens_models)):
        print(f'Model {(ensemble_index * num_ensembles) + (model_idx + 1)} {ens_models[model_idx]}')
        model.load_state_dict(torch.load(MODELS_PATH + ens_models[model_idx]))
        model.eval()
        tmp_probs = None
        y_true_class = []  # 1D array with labels in test set
        y_pred = []
        for inputs, labels in dataloader:
            y_true_class.extend(labels.data.cpu().numpy())
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            tmp_probs = np.concatenate(
                (tmp_probs, probs.data.cpu().numpy())) if tmp_probs is not None else probs.data.cpu().numpy()
        y_pred_probs_mult_ens[:, :, model_idx, ensemble_index] = tmp_probs
        y_true_class = np.array(y_true_class)

# compute uncertainties
unc_types = ['aleatoric', 'epistemic', 'total']
unc_method = ['entropy', 'variance', 'entropy_binarized']
y_unc_mult_ens = np.zeros((dataset_size, num_classes, len(unc_types), len(unc_method), num_ensembles))
for ensemble_index in range(num_ensembles):
    for m_idx, method in enumerate(unc_method):
        if method == 'entropy':
            y_unc_mult_ens[:, :, 0, m_idx, ensemble_index] = np.tile(
                aleatoric_uncertainty_entropy(y_pred_probs_mult_ens[:, :, :, ensemble_index]), (num_classes, 1)).T
            y_unc_mult_ens[:, :, 1, m_idx, ensemble_index] = np.tile(
                epistemic_uncertainty_entropy(y_pred_probs_mult_ens[:, :, :, ensemble_index]), (num_classes, 1)).T
            y_unc_mult_ens[:, :, 2, m_idx, ensemble_index] = np.tile(
                total_uncertainty_entropy(y_pred_probs_mult_ens[:, :, :, ensemble_index]), (num_classes, 1)).T
        if method == 'variance':
            y_unc_mult_ens[:, :, 0, m_idx, ensemble_index] = aleatoric_uncertainty_variance_label(
                y_pred_probs_mult_ens[:, :, :, ensemble_index])
            y_unc_mult_ens[:, :, 1, m_idx, ensemble_index] = epistemic_uncertainty_variance_label(
                y_pred_probs_mult_ens[:, :, :, ensemble_index])
            y_unc_mult_ens[:, :, 2, m_idx, ensemble_index] = total_uncertainty_variance_label(
                y_pred_probs_mult_ens[:, :, :, ensemble_index])
        if method == 'entropy_binarized':
            y_unc_mult_ens[:, :, 0, m_idx, ensemble_index] = np.tile(uncertainty_1vsall_entropy(
                y_pred_probs_mult_ens[:, :, :, ensemble_index], kind='au'), (num_classes, 1)).T
            y_unc_mult_ens[:, :, 1, m_idx, ensemble_index] = np.tile(uncertainty_1vsall_entropy(
                y_pred_probs_mult_ens[:, :, :, ensemble_index], kind='eu'), (num_classes, 1)).T
            y_unc_mult_ens[:, :, 2, m_idx, ensemble_index] = np.tile(uncertainty_1vsall_entropy(
                y_pred_probs_mult_ens[:, :, :, ensemble_index], kind='tu'), (num_classes, 1)).T

with open(f'{RESULTS_PATH}probs_multiple_ensembles.pkl', 'wb') as f:
    pickle.dump(y_pred_probs_mult_ens, f)

with open(f'{RESULTS_PATH}uncs_multiple_ensembles.pkl', 'wb') as f:
    pickle.dump(y_unc_mult_ens, f)

with open(f'{RESULTS_PATH}y_true_class.pkl', 'wb') as f:
    pickle.dump(y_true_class, f)

with open(f'{RESULTS_PATH}file_names.pkl', 'wb') as f:
    pickle.dump(file_names, f)

##########
# Evaluation
##########

# 1. compute standard metrics
ensemble_accs = []
for ens_idx in range(num_ensembles):
    overall_acc, acc, fpr, fnr, tpr, tnr = eval.compute_metrics(y_true_class, np.argmax(
        np.mean(y_pred_probs_mult_ens[:, :, :, ens_idx], axis=2), axis=1))
    ensemble_accs.append(overall_acc)
    print(f'Overall slice accuracy for ensemble {ens_idx}: {overall_acc}')
    print(f'Average per class slice accuracy {acc}, fpr: {fpr}, fnr: {fnr}, tpr: {tpr}, tnr: {tnr}')

print(f'Ensemble Accuracy: {np.mean(ensemble_accs)} +/- {np.std(ensemble_accs)}')

# 2. label wise uncertainty plot images
shutil.rmtree(f'{RESULTS_PATH}label_uncertainty/')
os.makedirs(f'{RESULTS_PATH}label_uncertainty/')
al_un_lbl = y_unc_mult_ens[:, :, 0, 1, 0]
ep_un_lbl = y_unc_mult_ens[:, :, 1, 1, 0]
to_un_lbl = y_unc_mult_ens[:, :, 2, 1, 0]
y_pred_class = np.argmax(np.mean(y_pred_probs_mult_ens[:, :, :, 0], axis=2), axis=1)
# identify images
number_images = 9
au_ind = np.argpartition(al_un_lbl.sum(axis=1), -number_images)[-number_images:]
eu_ind = np.argpartition(ep_un_lbl.sum(axis=1), -number_images)[-number_images:]
tu_ind = np.argpartition(to_un_lbl.sum(axis=1), -number_images)[-number_images:]

tu_neg_ind = np.argwhere((to_un_lbl[:, to_un_lbl.shape[1] - 1] < 0.01) & (ep_un_lbl.sum(axis=1) > 0.2))
tu_neg_ind = tu_neg_ind[:number_images, 0]

# plot images
for inds, name in zip([tu_ind, au_ind, eu_ind, tu_neg_ind], ['tu', 'au', 'eu', 'tu_neg']):
    print(f'{inds} {name}')
    for i, ind in enumerate(inds):
        # load images
        path = file_names[ind][0].split('/')
        file = path[-1].split('SUV_CT')
        cl = path[-2]
        pred = y_pred_class[ind]
        imgs = []
        imgs.append(
            scipy.ndimage.zoom(np.array(Image.open(f'/data/ct/{file[0]}CT{file[1]}')),
                               zoom=(1.4, 1, 1)))
        imgs.append(
            scipy.ndimage.zoom(np.array(Image.open(f'/data/suv/{file[0]}SUV{file[1]}')),
                               zoom=(1.4, 1, 1)))
        imgs.append(np.array(Image.open(f'/data/seg/{file[0]}SEG{file[1]}'))[:, :, 0])
        ind = [ind]
        plotter.plot_images_uncertainty(imgs, class_labels, al_un_lbl[ind], ep_un_lbl[ind], y_up=0.27,
                                        save_path=f'{RESULTS_PATH}label_uncertainty/image_label_uncertainty_{name}_{i}_class_{cl}_pred_{pred}.pdf')

# 3. accuracy rejection curve
# plot accuracy rejection curve
rand_order = np.zeros((y_true_class.shape[0], num_ensembles)).astype(int)
for idx in range(num_ensembles):
    rand_order[:, idx] = random.sample(range(0, y_true_class.shape[0]), y_true_class.shape[0])
plotter.plot_accuracy_rejection_curve(true_labels=y_true_class,
                                      pred_labels=np.argmax(np.mean(y_pred_probs_mult_ens[:, :, :, :], axis=2), axis=1),
                                      uncertainties=y_unc_mult_ens[:, 0, :, 0, :],
                                      rand_order=rand_order,
                                      save_path=f'{RESULTS_PATH}accuracy_rejection_curve.pdf')

plotter.plot_accuracy_rejection_curve(true_labels=y_true_class,
                                      pred_labels=np.argmax(np.mean(y_pred_probs_mult_ens[:, :, :, :], axis=2), axis=1),
                                      uncertainties=np.sum(y_unc_mult_ens, axis=1)[:, 0, [2, 0, 1], :],
                                      unc_labels=['$AU_{lent}$', "$AU_{ent}$", '$AU_{var}$'],
                                      rand_order=rand_order,
                                      save_path=f'{RESULTS_PATH}accuracy_rejection_curve_diff_measures_aleatoric.pdf')

plotter.plot_accuracy_rejection_curve(true_labels=y_true_class,
                                      pred_labels=np.argmax(np.mean(y_pred_probs_mult_ens[:, :, :, :], axis=2), axis=1),
                                      uncertainties=np.sum(y_unc_mult_ens, axis=1)[:, 1, [2, 0, 1], :],
                                      unc_labels=['$EU_{lent}$', "$EU_{ent}$", '$EU_{var}$'],
                                      rand_order=rand_order,
                                      save_path=f'{RESULTS_PATH}accuracy_rejection_curve_diff_measures_epistemic.pdf')

plotter.plot_accuracy_rejection_curve(true_labels=y_true_class,
                                      pred_labels=np.argmax(np.mean(y_pred_probs_mult_ens[:, :, :, :], axis=2), axis=1),
                                      uncertainties=np.sum(y_unc_mult_ens, axis=1)[:, 2, [2, 0, 1], :],
                                      unc_labels=['$TU_{lent}$', "$TU_{ent}$", '$TU_{var}$'],
                                      rand_order=rand_order,
                                      save_path=f'{RESULTS_PATH}accuracy_rejection_curve_diff_measures_total.pdf')
