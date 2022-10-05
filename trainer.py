from typing import Callable, Tuple, List

import torchvision

from utils import metrics
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import torchvision


class Trainer:
    def __init__(self, criterion: Callable,
                 accuracy: Tuple[int, ...] = (1, 5),
                 device: str = 'cpu',
                 patience: int = 20,
                 stop_on_acc: bool = True
                 ):
        self.criterion = criterion
        self.accuracy = accuracy
        self.device = device

        self.training_records = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
        self.training_records.index.name = 'epoch'

        self.patience = patience
        self.best_val_loss = np.inf
        self.best_val_acc = 0
        self.best_epoch = 0

        self.is_loss_lower = False
        self.patience_count = 0
        self.stop = False

    def train(self, epoch,
              model: torchvision.models,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              optimizer: torch.optim):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            batch_acc = metrics.accuracy(outputs, targets, (1, 5))
            train_acc += batch_acc[0]

        train_acc /= len(train_dataloader)
        train_acc = float(train_acc.cpu().numpy()[0])

        # validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_acc = metrics.accuracy(outputs, targets, (1, 5))
                val_acc += batch_acc[0]
        val_acc /= len(val_dataloader)
        val_acc = float(val_acc.cpu().numpy()[0])

        lr = optimizer.state_dict()['param_groups'][0]['lr']

        epoch_record = {'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc, 'lr': lr}
        epoch_record = dict(epoch_record)
        self.training_records.append(epoch_record, ignore_index=True)
        if val_acc > self.best_val_acc:
            self.patience_count = 0
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.is_loss_lower = True
        else:
            self.patience_count += 1
            self.is_loss_lower = False
            if self.patience and self.patience_count >= self.patience:
                self.stop = True

        return train_loss, train_acc, val_loss, val_acc, lr
