import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.ensemble as se
import torchvision.models as tm
import sklearn.tree as st

import utils

EPS = 1e-10

class RandomForest(se.RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_liks = np.zeros(self.n_estimators)

    def set_norm_liks(self, x, y):
        for i in range(self.n_estimators):
            y_pred = self.estimators_[i].predict_proba(x)
            lik = np.sum(np.log(y_pred[np.arange(y.shape[0]), y] + EPS))
            lik = -1 / lik
            self.norm_liks[i] = lik
        self.norm_liks /= self.norm_liks.max()

    def predict(self, x, alpha=None):
        preds = np.empty((x.shape[0], self.n_classes_, self.n_estimators))
        for i in range(self.n_estimators):
            if alpha is None or self.norm_liks[i] >= alpha:
                preds[:, :, i] = self.estimators_[i].predict_proba(x)
        return preds


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class Ensemble(nn.Module):
    def __init__(self, model, num_models):
        super().__init__()
        if model == "cnn":
            self.members = [CNN() for _ in range(num_models)]
        elif model == "resnet":
            self.members = []
            for _ in range(num_models):
                resnet = tm.resnet18()
                resnet.fc = nn.Linear(512, 10)
                # resnet = tm.resnet50()
                # resnet.fc = nn.Linear(2048, 10)
                self.members.append(resnet)
        else:
            raise ValueError("Invalid model name")
        self.norm_liks = torch.zeros(num_models)

    def set_norm_liks(self, loader):
        for i in range(len(self.members)):
            for inputs, targets in loader:
                outputs = F.softmax(self.members[i](inputs), dim=1)
                self.norm_liks[i] += torch.sum(torch.log(
                    outputs[torch.arange(targets.shape[0]), targets]))
            self.norm_liks[i] = -1 / self.norm_liks[i]
        self.norm_liks /= self.norm_liks.max()

    def forward(self, x, alpha=None):
        preds = torch.empty((x.shape[0], 10, len(self.members)))
        for i in range(len(self.members)):
            if alpha is None or self.norm_liks[i] >= alpha:
                preds[:, :, i] = F.softmax(self.members[i](x), dim=1)
        return preds




