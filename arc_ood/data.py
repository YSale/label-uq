import numpy as np
import sklearn
import sklearn.datasets as sd
import torchvision.transforms as T
import torch
import torchvision
from torch.utils.data import DataLoader
import sklearn.preprocessing as sprep
import PIL
import os
import datasets
from tqdm import tqdm
import uncertainty as unc
import utils
import pandas as pd

def get_data(dataset: str, size = 1.0, flatten = True, portion = 1.0, train_set = ''):
    if dataset == "covtype":
        data = sklearn.datasets.fetch_covtype()
        # data = sklearn.datasets.fetch_openml(data_id=150, as_frame=False, parser="auto")
        x, y = data.data, data.target
        # y = y.astype(int) - 1
        y = y - 1
    elif dataset == "poker":
        data = sklearn.datasets.fetch_openml(data_id=1569, as_frame=False, parser="auto")
        x, y = data.data, data.target
        y = y.astype(int) - 1
    elif dataset == "digits":
        data = sklearn.datasets.load_digits()
        x, y = data.data, data.target
    elif dataset == "newsgroup":
        data = sklearn.datasets.fetch_20newsgroups_vectorized()
        x, y = data.data, data.target
    elif dataset == "yeast":
        data = sklearn.datasets.fetch_openml(data_id=181, as_frame=False, parser="auto")
        x, y = data.data, data.target
        y = sprep.LabelEncoder().fit_transform(y)
    elif dataset == "kddcup99":
        data = sklearn.datasets.fetch_kddcup99()
        le = sprep.LabelEncoder().fit_transform(data.target)
        x, y = data.data, data.target
    elif dataset == "wine": # to easy
        data = sklearn.datasets.load_wine()
        x, y = data.data, data.target
    elif dataset == "diabetes":
        df = sklearn.datasets.fetch_openml(data_id=43063).data
        x = df.loc[:, df.columns != "class"].to_numpy()
        y = df.loc[:, "class"].to_numpy()
        x = np.array(x)
        x = sprep.scale(x)
        y[y == "tested_positive"] = 1
        y[y == "tested_negative"] = 0
        y = y.astype(int)
    elif dataset == "parkinsons":
        x, y = sklearn.datasets.fetch_openml(data_id=1488, return_X_y=True)
        x = x.to_numpy()
        y = y.to_numpy()
        y = y.astype(int) - 1

    elif dataset == "mushroom":
        x, y = sklearn.datasets.fetch_openml(data_id=24, return_X_y=True)
        x = x.to_numpy()
        y = y.to_numpy()
        print(x.shape)
        y[y == "p"] = 1
        y[y == "e"] = 0
    elif dataset == "toy":
        data = sd.make_classification(n_samples=10000, n_features=40, n_informative=40, n_redundant=0, n_classes=10, n_clusters_per_class=1)
        x,y = data[0], data[1]
    # elif dataset == "mnist":
    #     train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    #     test = torchvision.datasets.MNIST(root="./data", train=False, download=True)
    #     return train, test
    elif dataset == "mnist":
        if flatten:
            train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                                  transform=T.Compose([T.ToTensor(), T.Resize(int(28*size)), lambda x: x.view(-1)]))
            test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                                 transform=T.Compose([T.ToTensor(), T.Resize(int(28*size)), lambda x: x.view(-1)]))
        else:
            train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                               transform=T.Compose(
                                                   [T.ToTensor(), T.Resize(int(28 * size))]))
            test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                              transform=T.Compose(
                                                  [T.ToTensor(), T.Resize(int(28 * size))]))
        if portion < 1.0:
            train = torch.utils.data.Subset(train, np.random.choice(len(train), int(len(train)*portion), replace=False))
            test = torch.utils.data.Subset(test, np.random.choice(len(test), int(len(test)*portion), replace=False))
        train_loader = DataLoader(train, batch_size=64, shuffle=False)
        test_loader = DataLoader(test, batch_size=64, shuffle=False)
        x = train_loader
        y = test_loader
    elif dataset == "fmnist":
        if flatten:
            train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True,
                                                  transform=T.Compose([T.ToTensor(), T.Resize(int(28*size)), lambda x: x.view(-1)]))
            test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True,
                                                 transform=T.Compose([T.ToTensor(), T.Resize(int(28*size)), lambda x: x.view(-1)]))
        else:
            train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True,
                                               transform=T.Compose(
                                                   [T.ToTensor(), T.Resize(int(28 * size))]))
            test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True,
                                              transform=T.Compose(
                                                  [T.ToTensor(), T.Resize(int(28 * size))]))
        # print(train.data[0])
        train_loader = DataLoader(train, batch_size=64, shuffle=False)
        test_loader = DataLoader(test, batch_size=64, shuffle=False)
        x = train_loader
        y = test_loader
    elif dataset == "kmnist":
        if flatten:
            train = torchvision.datasets.KMNIST(root="./data", train=True, download=True,
                                                      transform=T.Compose([T.ToTensor(), T.Resize(int(28 * size)),
                                                                           lambda x: x.view(-1)]))
            test = torchvision.datasets.KMNIST(root="./data", train=False, download=True,
                                                     transform=T.Compose([T.ToTensor(), T.Resize(int(28 * size)),
                                                                          lambda x: x.view(-1)]))
        else:
            train = torchvision.datasets.KMNIST(root="./data", train=True, download=True,
                                                      transform=T.Compose(
                                                          [T.ToTensor(), T.Resize(int(28 * size))]))
            test = torchvision.datasets.KMNIST(root="./data", train=False, download=True,
                                                     transform=T.Compose(
                                                         [T.ToTensor(), T.Resize(int(28 * size))]))
        train_loader = DataLoader(train, batch_size=64, shuffle=False)
        test_loader = DataLoader(test, batch_size=64, shuffle=False)
        x = train_loader
        y = test_loader
    elif dataset == "notmnist":
        classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        dir = "./data/notMNIST/"
        data = []
        targets = []
        for i, c in enumerate(classes):
            list = os.listdir(dir + c)
            if ".DS_Store" in list:
                list.remove(".DS_Store")
            for l in list:
                data.append(np.array(PIL.Image.open(dir + c + "/" + l)))
                targets.append(i)
        data = np.array(data)
        data = np.expand_dims(data, axis=1)
        targets = np.array(targets)
        dataset = torch.utils.data.dataset.TensorDataset(torch.Tensor(data), torch.Tensor(targets))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        return None, loader
    elif dataset == "cifar":
        if flatten:
            raise ValueError("Cannot flatten CIFAR")
        train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                                transform=T.Compose([
                                                 T.RandomCrop(32, padding=4),
                                                 T.RandomHorizontalFlip(),
                                                 T.ToTensor(),
                                                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                             ]))
        test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                                 transform=T.Compose([
                                                 T.ToTensor(),
                                                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                             ]))
        train_loader = DataLoader(train, batch_size=64, shuffle=False)
        test_loader = DataLoader(test, batch_size=64, shuffle=False)
        x = train_loader
        y = test_loader
    elif dataset == "lsun":
        if flatten:
            raise ValueError("Cannot flatten LSUN")
        train = torchvision.datasets.LSUN(root="./data", classes="train", download=True,
                                                  transform=T.ToTensor())
        test = torchvision.datasets.LSUN(root="./data", classes="test", download=True,
                                                 transform=T.ToTensor())
        train_loader = DataLoader(train, batch_size=64, shuffle=False)
        test_loader = DataLoader(test, batch_size=64, shuffle=False)
        x = train_loader
        y = test_loader
    elif dataset == "svhn":
        if flatten:
            raise ValueError("Cannot flatten SVHN")
        train = torchvision.datasets.SVHN(root="./data", split="train", download=True,
                                                  transform=T.ToTensor())
        
        if train_set == 'cifar':
            transforms = T.Compose([T.ToTensor(),
                                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        else:
            transforms = T.ToTensor()
        test = torchvision.datasets.SVHN(root="./data", split="test", download=True,
                                                 transform=transforms
                                            )
        train_loader = DataLoader(train, batch_size=64, shuffle=False)
        test_loader = DataLoader(test, batch_size=64, shuffle=False)
        x = train_loader
        y = test_loader
    elif dataset == "tinyimagenet":
        if flatten:
            raise ValueError("Cannot flatten Tiny-ImageNet")
        data = datasets.load_dataset('Maysee/tiny-imagenet', split='valid')

        x = []
        y = []
        for i in tqdm(range(len(data))):
            d = np.asarray(data[i]['image'])
            l = np.asarray(data[i]['label'])
            if d.ndim == 3:
                x.append(np.swapaxes(d, 0, 2))
                y.append(l)
        x = torch.tensor(np.array(x)/255).float()
        y = torch.tensor(np.array(y)).float()
        dataset = torch.utils.data.dataset.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        return None, loader
    elif dataset == "cifar2":
        data = np.load("./data/cifar102_test.npz")
        x = data["images"]
        x = np.swapaxes(x, 1, 3)
        y = data["labels"]
        transform=T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #x = torch.stack([transform(xi) for xi in x])
        x= torch.Tensor(x)
        if train_set == 'cifar':
            x = transform(x)
        dataset = torch.utils.data.dataset.TensorDataset(x, torch.Tensor(y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        return None, loader
    else:
        raise ValueError("Invalid dataset name")
    return x, y


def get_unc_scores(data_str, seed=1, score_functions = [unc.total_uncertainty_variance, unc.total_uncertainty_entropy], score_names = ['TU_var', 'TU_ent']):
    _, x = get_data(data_str, flatten = False) 
    model = torch.load(f'./checkpoints/{data_str}_{seed}.pt',map_location=torch.device('mps'))
    for member in model.members:
        member.to('mps')
        member.eval()
    probs, y = utils.torch_get_outputs(model, x)
    probs = probs.detach().numpy()
    y = y.detach().numpy()
    preds = np.argmax(np.mean(probs, axis=2), axis=1)
    diction = {'correct': y==preds}
    for score_fun, s_name in zip(score_functions, score_names):
        diction[s_name] = score_fun(probs)
    unc_frame = pd.DataFrame(diction)
    return unc_frame


