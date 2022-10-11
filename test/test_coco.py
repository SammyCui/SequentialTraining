import argparse
import os
import sys
sys.path.append('/u/erdos/students/xcui32/SequentialTraining')
sys.path.append('/u/erdos/students/xcui32')
sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages')
sys.path.append('/u/erdos/students/xcui32/.local/lib/python3.6/site-packages')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from SequentialTraining.trainer import Trainer
from SequentialTraining.utils.coco_utils import get_big_coco_classes
from SequentialTraining.utils.data_utils import GenerateBackground
from SequentialTraining.loader import CIFARLoader
from SequentialTraining.datasets_legacy import CIFAR10Dataset, COCODataset
from SequentialTraining.utils import metrics
import torchvision

torch.autograd.detect_anomaly(True)

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device: ', device)
# Data
print('==> Preparing data..')
transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

dataset_COCO = {'train_annotation_path': '/u/erdos/cnslab/coco/annotations/classification_train2017.json',
                'train_image_path': '/u/erdos/cnslab/coco/train',
                'test_annotation_path': '/u/erdos/cnslab/coco/annotations/classification_test2017.json',
                'test_image_path': '/u/erdos/cnslab/coco/test'}
train_annotation_path = dataset_COCO['train_annotation_path']
test_annotation_path = dataset_COCO['train_annotation_path']
num_classes = 40
min_image_per_class = 580
lr_patience = 5
min_lr = 0.00001
all_regimens = ['stb_endsame', 'bts_startsame', 'llo', 'random_oneseq', 'random1', 'single']
cls_to_use = get_big_coco_classes(150, path_to_json=train_annotation_path if train_annotation_path
else dataset_COCO['train_annotation_path'][0],
                                  min_image_per_class=min_image_per_class, num_classes=num_classes)
optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9}
scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': lr_patience, 'min_lr': min_lr}
'/u/erdos/cnslab/coco/annotations/classification_train2017.json'

train_dataset = COCODataset(root=dataset_COCO['train_image_path'], path_to_json=train_annotation_path,
                            transform=transform,
                            input_size=(150, 150), size=1, resize_method='long', cls_to_use=cls_to_use,
                            min_image_per_class=580,
                            max_image_per_class=800, num_classes=num_classes)

num_samples = len(train_dataset)

test_dataset = COCODataset(root=dataset_COCO['test_image_path'], path_to_json=test_annotation_path, transform=transform,
                           input_size=(150, 150), size=1, resize_method='long', cls_to_use=cls_to_use,
                           min_image_per_class=580,
                           max_image_per_class=800, num_classes=num_classes)

n_folds = 5
val_size = 1 / n_folds
cv_indices = list(range(num_samples))
np.random.seed(40)
np.random.shuffle(cv_indices)
split = int(np.floor(val_size * num_samples))
optimizer_object = torch.optim.SGD
criterion_object = torch.nn.CrossEntropyLoss
scheduler_object = torch.optim.lr_scheduler.ReduceLROnPlateau
for fold in range(1):
    split1 = int(np.floor(fold * split))
    split2 = int(np.floor((fold + 1) * split))
    val_idx = cv_indices[split1:split2]
    train_idx = np.append(cv_indices[:split1], cv_indices[split2:])
    train_idx = train_idx.astype('int32')
    model = torchvision.models.resnet18(num_classes=len(train_dataset.classes))
    model.to(device)
    optimizer = optimizer_object(model.parameters(), **optimizer_kwargs)
    scheduler = scheduler_object(optimizer, **scheduler_kwargs)

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True,
        sampler=SubsetRandomSampler(train_idx))
    val_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True,
        sampler=SubsetRandomSampler(val_idx))
    trainer = Trainer(criterion=criterion_object(), patience=20, device=device)
    for epoch in range(10):
        train_loss, train_acc, val_loss, val_acc, lr = trainer.train(epoch, model, train_dataloader,
                                                                     val_dataloader, optimizer, max_norm=None)

        print(
            'Epoch [{}/{}] Training Loss: {:.4f} Training Acc: {:.3f} Val Loss: {:.4f} Val Acc: {:.3f} lr: {:.4f}'
                .format(epoch + 1, 10, train_loss, train_acc, val_loss, val_acc, lr))
