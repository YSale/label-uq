import torch
from torchvision import transforms as T

import data_loading as dl
import trainer
import models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'resnet50'
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

########################
# Data Preparation
########################

# get dataloader
dataloaders = dl.get_dataloaders(data_path='/data/', batch_size=50,
                                 train_transform=supervise_train_transform,
                                 test_transform=supervise_test_transform)
class_names = dataloaders['test'].dataset.classes
num_classes = len(class_names)

#######################
# Model Training
#######################

# get model
model, criterion, optimizer, lr_scheduler = models.get_resnet50(num_classes)

# train model
best_model = trainer.train_supervised_neural_network(model=model, model_name=MODEL_NAME,
                                                     results_path=RESULTS_PATH, dataloaders=dataloaders,
                                                     criterion=criterion, optimizer=optimizer, scheduler=lr_scheduler,
                                                     num_epochs=40)
