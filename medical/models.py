import torch
from torch import nn, optim
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_resnet50(num_classes=None):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet50 = models.resnet50(weights=weights)
    if num_classes is not None:
        resnet50.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    resnet50 = resnet50.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(resnet50.parameters(), lr=0.001, weight_decay=5e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return resnet50, criterion, optimizer, exp_lr_scheduler
