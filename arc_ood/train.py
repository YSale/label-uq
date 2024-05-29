import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tm
import torchvision.transforms as T
import random
import numpy as np
import models as mds
from tqdm import tqdm

import os

DEVICE = "mps"

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.mps.manual_seed(args.seed)

    if args.data == "mnist":
        train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                           transform=T.ToTensor())
        test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                          transform=T.ToTensor())
    if args.data == "fmnist":
        train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True,
                                           transform=T.ToTensor())
        test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True,
                                          transform=T.ToTensor())
    elif args.data == "cifar":
        train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                             transform=T.Compose([
                                                 T.RandomCrop(32, padding=4),
                                                 T.RandomHorizontalFlip(),
                                                 T.ToTensor(),
                                                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                             ]))
        test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                            transform=T.ToTensor())
    # train, val = torch.utils.data.random_split(train, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)
    if args.data == "mnist" or args.data == "fmnist":
        if args.type == 'ensemble':
            model = mds.Ensemble("cnn", 5)
            for member in model.members:
                member = member.to(DEVICE)
        elif args.type =='diri':
            model = mds.CNN_diri()
            model.to(DEVICE)
    elif args.data == "cifar":
        if args.type == 'ensemble':
            model = mds.Ensemble("resnet", 5).to(DEVICE)
            for member in model.members:
                member = member.to(DEVICE)
    if args.type == 'ensemble':
        criterion = nn.CrossEntropyLoss()
    elif args.type == 'diri':
        criterion = mds.crit_ml

    if args.type == 'ensemble':
        for i in tqdm(range(len(model.members)), desc="Member"):
            optimizer = optim.SGD(model.members[i].parameters(), lr=0.1, momentum=0.9, weight_decay=10**-4)
            # optimizer = optim.Adam(model.members[i].parameters())
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)
            for epoch in tqdm(range(args.epochs), desc="Epochs"):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model.members[i](inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                # correct = 0
                # total = 0
                # for inputs, targets in test_loader:
                #     inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                #     outputs = model.members[i](inputs)
                #     correct += torch.sum(torch.argmax(outputs, dim=1) == targets)
                #     total += targets.shape[0]
                # acc = correct / total

                # print("Epoch", epoch, "Accuracy", acc)
    
    elif args.type == 'diri':
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=10**-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)
        for epoch in tqdm(range(args.epochs), desc="Epochs"):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()     

    # model = model.to("cpu")
    # for member in model.members:
    #     member = member.to("cpu")
    # model.set_norm_liks(train_loader)
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')
    torch.save(model, f"./checkpoints/{str(args.data)}_{str(args.type)}_{str(args.seed)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--type", type=str, default='ensemble')
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
