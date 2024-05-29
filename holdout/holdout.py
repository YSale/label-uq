import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
from tqdm import tqdm
from data import get_data, get_probs
import models as mds
import unc_label as unc
import random
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import argparse

DEVICE = "cuda"

def train_ensemble(model, train_loader, epochs, optims, schedulers=None, hold_inds = None, add_class = None, full_train_set = None):
    criterion = nn.CrossEntropyLoss()
    if hold_inds is not None:
        train_set = train_loader.dataset
        num_add = int(len(train_set)/10)
        train_indices = [idx for i, idx in enumerate(train_set.indices) if train_set[i][1] != add_class] 
    for i in range(len(model.members)):
        if schedulers is not None:
            scheduler = schedulers[i]
        optimizer = optims[i]
        model.members[i].to(DEVICE)
        tqdm.write(f"\nTraining ensemble member {i+1}")
        for epoch in tqdm(range(epochs)):
            if hold_inds is not None:
                add_inds =  random.sample(hold_inds, num_add) 
                train_set2 = Subset(full_train_set, train_indices + add_inds)
                train_loader = DataLoader(train_set2, batch_size=train_loader.batch_size, shuffle=True)

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model.members[i](inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if schedulers is not None:
                scheduler.step()
    return model

def get_new_ensemble(data, num_members):
    if(data not in ['cifar', 'svhn']):
        ensemble = mds.Ensemble("cnn", num_members)
    else:
        ensemble = mds.Ensemble("resnet", num_members)
    return ensemble


def main(args):
    if args.seed != None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    #print(f'data: {args.data}, hold_rate: {args.hold_rate}, add_rate: {ADD_RATE}, train_new: {args.train_new}')
    config = vars(args)
    print(config)
    eus_before = []
    eus_after = []
    train_loader_full, test_loader = get_data(args.data, flatten=False)
    for run in tqdm(range(args.runs)):
        full_train_dataset = train_loader_full.dataset
        holdout_size = int(args.hold_rate * len(full_train_dataset))
        train_size = len(full_train_dataset) - holdout_size
        train_dataset, train_holdout_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, holdout_size])
        train_loader = DataLoader(train_dataset, batch_size=train_loader_full.batch_size, shuffle=True)
        ensemble = get_new_ensemble(args.data, args.num_members)

        if(args.data not in ['cifar', 'svhn']):
            optims = [optim.Adam(m.parameters()) for m in ensemble.members]
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
            schedulers = None
        else:
            optims = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) for m in ensemble.members]
            schedulers = [optim.lr_scheduler.MultiStepLR(optims[i], milestones=[20, 25], gamma=0.1) for i in range(len(ensemble.members))]
        print('\nTraining on set of size:', train_size)
        ensemble = train_ensemble(ensemble, train_loader, args.epochs, optims, schedulers)
        
        probs, labels= get_probs(test_loader, ensemble, DEVICE)
        eus = unc.epistemic_uncertainty_variance(probs)
        eu_mean = eus.mean(0)
        acc = (np.argmax(probs.mean(-1), -1) == labels).mean()
        print('\neu before:', eu_mean.round(5))
        eus_before.append(eus)
        
        class_max = np.argmax(eu_mean)
        print(f'Class with highest eu: {class_max} with {eu_mean[class_max]:.5f}')
        print(f'{acc=:.3f}')
        hold_out_inds = train_holdout_dataset.indices
        hold_inds = [idx for idx in hold_out_inds if full_train_dataset[idx][1] == class_max]

        if args.train_new:
            ensemble = get_new_ensemble(args.data, args.num_members)
            if(args.data not in ['cifar', 'svhn']):
                optims = [optim.Adam(m.parameters()) for m in ensemble.members]
                #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
                schedulers = None
            else:
                optims = [optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) for m in ensemble.members]
                schedulers = [optim.lr_scheduler.MultiStepLR(optims[i], milestones=[20, 25], gamma=0.1) for i in range(len(ensemble.members))]
            #print(f'Train new on: {len(train_holdout_dataset)} observations with class {class_max} added')
            print('train new')
        else:
            print('continue training')
        ensemble.train()
        ensemble = train_ensemble(ensemble, train_loader, args.epochs_hold, optims, schedulers, hold_inds, class_max, full_train_dataset)
        probs_after, labels_after = get_probs(test_loader, ensemble, DEVICE)
        acc_after = (np.argmax(probs_after.mean(-1),-1) == labels_after).mean()
        eus = unc.epistemic_uncertainty_variance(probs_after)
        eu_mean_after = eus.mean(0)
        print('\neu after:', eu_mean_after.round(5))
        print(f'Class with highest eu: {class_max} with {eu_mean_after[class_max]:.5f}')
        print(f'{acc_after=:.3f}')
        eus_after.append(eus)

    eus_before = np.array(eus_before)
    eus_after = np.array(eus_after)

    path = f'results/results_{args.data}'
    np.savez(path+'.npz', eus_before = eus_before, eus_after = eus_after, **config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--epochs_hold", type=int)
    parser.add_argument("--hold_rate", type=float)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--num_members", type=int)
    args = parser.parse_args()
    main(args)


    