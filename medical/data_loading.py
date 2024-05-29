import os
import torch
import numpy as np
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset


def get_dataloaders(data_path, batch_size, train_transform, test_transform, balanced_sampling=False,
                    subset_percentage=None):
    image_datasets = {'train': ImageFolder(os.path.join(data_path, 'train'), train_transform),
                      'test': ImageFolder(os.path.join(data_path, 'test'), test_transform)}

    if subset_percentage:
        indices = random.sample(range(0, len(image_datasets['train'])),
                                int(subset_percentage * len(image_datasets['train'])))
        image_datasets['train'] = Subset(image_datasets['train'], indices)

    if balanced_sampling:
        y_train_tuples = image_datasets['train'].imgs
        y_train = [class_idx for path, class_idx in y_train_tuples]
        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=sampler,
                            shuffle=train_shuffle,
                            num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False,
                           num_workers=4)}
    return dataloaders
