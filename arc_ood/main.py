import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms

import utils
from utils import accuracy
import sklearn.metrics as sm
import torch.optim as optim
from tqdm import tqdm
import time
from data import get_data
import models as mds
import os
import experiments
import sklearn.model_selection as sms
import plot
import uncertainty as unc
import config
import sklearn.tree as st
import random
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchmetrics as tm
if config.SEED != None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

DEVICE = "mps"

def train_ensemble(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    for i in range(len(model.members)):
        model.members[i].to(DEVICE)
        print(f"Training ensemble member {i+1}")
        if(config.DATA not in ['cifar', 'svhn']):
            optimizer = optim.Adam(model.members[i].parameters())
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        else:
            optimizer = optim.SGD(model.members[i].parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)
        for epoch in tqdm(range(config.EPOCHS)):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model.members[i](inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if(config.DATA in ['cifar', 'svhn']):
                scheduler.step()
    return model


def main():
    if config.EXP == "accrej":
        x, y = get_data(config.DATA, flatten=False)
        accs = np.empty((config.RUNS, 4, 50))
        for run in tqdm(range(config.RUNS)):
            model_path = f"./checkpoints/{config.DATA}_{run+1}.pt"
            if isinstance(x, torch.utils.data.DataLoader):
                if(config.DATA not in ['cifar', 'svhn']):
                    ensemble = mds.Ensemble("cnn", config.NUM_MEMBERS)
                else:
                    ensemble = mds.Ensemble("resnet", config.NUM_MEMBERS)
                print("Loading model")
                if os.path.isfile(model_path):
                    print("Using saved model")
                    ensemble = torch.load(model_path, map_location=torch.device('cpu')) ## careful
                else:
                    print("Training new model")
                    ensemble = train_ensemble(ensemble, x)
                    if not os.path.exists('./checkpoints/'):
                        os.makedirs('./checkpoints/')
                    torch.save(ensemble, model_path)
                for member in ensemble.members:
                    member.to('cpu')
                    member.eval()
                acc = experiments.accuracy_rejection(ensemble, y, None, unc_method=config.ARC_MEASURE)
            else:
                x_train, x_test, y_train, y_test = sms.train_test_split(x, y, test_size=config.TEST_SIZE)
                rf = mds.RandomForest(n_estimators=config.NUM_MEMBERS, max_depth=config.MAX_DEPTH, bootstrap=True,
                                      criterion="entropy")
                rf.fit(x_train, y_train)
                rf.set_norm_liks(x_train, y_train)
                print(rf.norm_liks)
                print("RandomForest accuracy", sm.accuracy_score(y_test, rf.predict(x_test).mean(axis=2).argmax(axis=1)))
                acc = experiments.accuracy_rejection(rf, x_test, y_test, "canonical")
                acc_base = experiments.accuracy_rejection(rf, x_test, y_test, "baseline")
            accs[run, :, :] = acc
        if not os.path.exists('./output/'):
            os.makedirs('./output/')
        np.save(f"./output/{config.EXP}_{config.DATA}_{config.ARC_MEASURE}.npy", accs)

    elif config.EXP == "ood":
        aurocs = np.empty(config.RUNS)
        fprs = []
        tprs = []
        train_loader, test_loader = get_data(config.DATA, flatten=False)
        for run in tqdm(range(config.RUNS)):
            ensemble = mds.Ensemble("cnn", config.NUM_MEMBERS)
            ensemble = torch.load(f"checkpoints/{config.DATA}_{run+1}.pt")
            for member in ensemble.members:
                member.eval()
            _, test_loader_ood = get_data(config.OOD_DATA, flatten=False, train_set=config.DATA)
            u_id, u_ood, auroc, fpr, tpr = experiments.out_of_distribution(ensemble, test_loader, test_loader_ood, config.OOD_MEASURE)
            aurocs[run] = auroc
            fprs.append(fpr)
            tprs.append(tpr)
            print(aurocs)
        
        fpr_grid = np.linspace(0, 1, 100)

        # Interpolate the TPR values for each model
        tpr_interp = []
        for fpr, tpr in zip(fprs, tprs):
            tpr_interp.append(np.interp(fpr_grid, fpr, tpr))

        # Compute the mean and standard deviation of the TPR values
        tpr_mean = np.mean(tpr_interp, axis=0)
        tpr_std = np.std(tpr_interp, axis=0)

        # Compute the AUC of the average ROC curve
        auc_mean = np.trapz(tpr_mean, fpr_grid)

        # Plot the average ROC curve with error bars
        plt.figure()
        plt.plot(fpr_grid, tpr_mean, label=f"Average ROC curve (AUC = {auc_mean:.2f})")
        plt.fill_between(fpr_grid, tpr_mean - tpr_std, tpr_mean + tpr_std, alpha=0.2, label="Standard deviation")
        for i, auc in enumerate(aurocs):
            plt.plot(fprs[i], tprs[i], linestyle="--", label=f"Model {i+1} (AUC = {auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Average ROC curve of an ensemble of models")
        plt.legend()
        plt.show()
        print(f"AUROC: mean={np.round(np.mean(aurocs), 3)}, std={np.round(np.std(aurocs), 3)}")

    
if __name__ == '__main__':
    print("RUNNING EXPERIMENT WITH FOLLOWING CONFIG")
    print("================================================")
    conf = vars(config)
    for key in conf.keys():
        if not key.startswith("__"):
            print(key, conf[key])
    print("================================================")
    main()
