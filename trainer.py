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
                 patience: int = 20
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
        train_loss /= len(train_dataloader)
        train_loss = float(train_loss.cpu().numpy()[0])

        model.eval()
        val_loss = 0
        val_acc = 0

        # validation
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
        self.training_records.append(epoch_record)
        if val_loss < self.best_val_loss:
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


class Loop:

    def __init__(self, regimen: List,
                 model_name: str,
                 num_classes: int,
                 total_epochs: int,
                 optimizer_object: Callable,
                 criterion_object: Callable = nn.CrossEntropyLoss,
                 scheduler_object: Callable = None,
                 early_stopping_patience: int = 20,
                 reset_lr: float = None,
                 max_norm: int = None,
                 device: str = 'cpu',
                 optim_kwargs: dict = None,
                 scheduler_kwargs: dict = None,
                 save_progress_ckpt: bool = False,
                 save_result_ckpt: bool = False,
                 ckpt_dir: str = None
                 ):
        """

        :param total_epochs:
        :param device:
        :param num_classes:
        :param model_name:
        :param regimen: a list of regimens, as returned from the regimens.py get_train_dataloaders
        :param optimizer_object:
        :param criterion_object:
        :param scheduler_object:
        :param early_stopping_patience:
        :param reset_lr:
        :param max_norm: max num for gradient clipping
        :param optim_kwargs: a dictionary for optimizer kwargs
        :param scheduler_kwargs: a dictionary for scheduler kwargs
        :return:
        """

        self.regimen = regimen
        self.model_name = model_name
        self.num_classes = num_classes
        self.total_epochs = total_epochs
        self.optimizer_object = optimizer_object
        self.criterion_object = criterion_object
        self.scheduler_object = scheduler_object
        self.early_stopping_patience = early_stopping_patience
        self.reset_lr = reset_lr
        self.max_norm = max_norm
        self.device = device
        self.optim_kwargs = optim_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.save_progress_ckpt = save_progress_ckpt
        self.save_result_ckpt = save_result_ckpt
        self.ckpt_dir = ckpt_dir

    def run(self):

        criterion = self.criterion_object()
        records = []

        for fold, sequence in enumerate(self.regimen):

            model_best_states = {}
            record_dict = {}
            print(f'Fold: {fold}')
            for sequence_name, sequence_dataloaders in sequence:
                # e.g. sequence_name: "[0.8, 1]"
                # sequence_dataloaders: [(train_dataloader_0.8, val_dataloader_0.8), (train_dataloader_1, val_dataloader_1)]
                print(f'==> Training {self.model_name} on {sequence_name}')
                if self.model_name == 'inception_v3' or self.model_name == 'googlenet':
                    model = eval('torchvision.models.' + self.model_name + f'(num_classes={self.num_classes}, aux_logits=False)')
                else:
                    model = eval('torchvision.models.' + self.model_name + f'(num_classes={self.num_classes})')
                model = model.to(self.device)
                epochs_per_size = int(np.ceil(self.total_epochs / len(sequence_dataloaders)))
                sequence_list = eval(sequence_name)  # [0.6, 0.8, 1]
                record_list = []
                for seq_idx, (train_dataloader, val_dataloader) in enumerate(sequence_dataloaders):

                    if str(sequence_list[:seq_idx + 1]) in model_best_states:
                        model.load_state_dict(model_best_states[str(sequence_list[:seq_idx + 1])])
                        record_list = record_list + record_dict[str(sequence_list[:seq_idx + 1])]
                        print(f"Sequence {sequence_list[:seq_idx + 1]} already in state dictionary, jumped")

                    else:

                        optimizer = self.optimizer_object(model.parameters(), **self.optim_kwargs)
                        if self.reset_lr and seq_idx > 0:
                            for g in optimizer.param_groups:
                                g['lr'] = self.reset_lr
                        if self.scheduler_object:
                            scheduler = self.scheduler_object(optimizer, **self.scheduler_kwargs)
                        if str(sequence_list[:seq_idx]) in model_best_states:
                            model.load_state_dict(model_best_states[str(sequence_list[:seq_idx])])
                            print(f'Loaded best state dict for {str(sequence_list[:seq_idx])}')

                        elif seq_idx == 0:
                            pass
                        else:
                            raise ValueError(
                                "Uh Something went wrong... the previous sequence is not in the state "
                                "dict...")

                        print(f'==> Current group: {sequence_list[seq_idx]}')
                        trainer = Trainer(criterion=criterion, device=self.device)
                        for epoch in range(epochs_per_size):

                            train_loss, train_acc, val_loss, val_acc, lr = trainer.train(epoch, model, train_dataloader,
                                                                                         val_dataloader, optimizer)

                            print(
                                'Epoch [{}/{}] Training Loss: {:.4f} Training Acc: {:.3f} Val Loss: {:.4f} Val Acc: {:.3f} lr: {:.4f}'
                                    .format(epoch + 1, epochs_per_size, train_loss, train_acc, val_loss, val_acc, lr))

                            if trainer.is_loss_lower:
                                model_best_states[str(sequence_list[:seq_idx + 1])] = model.state_dict()
                                if self.save_progress_ckpt:
                                    fold_dir = os.path.join(self.ckpt_dir, f'fold_{fold}')
                                    if not os.path.isdir(fold_dir):
                                        os.mkdir(fold_dir)
                                    torch.save(model.state_dict(),
                                               os.path.join(fold_dir, f'{str(sequence_list[:seq_idx + 1])}.pt'))

                            else:
                                if trainer.stop:
                                    print(" --- Early Stopped ---")
                                    break
                            if self.scheduler_object:
                                scheduler.step(val_loss)

                        record_list.append(trainer)

                        print(
                            f"Group: {sequence_list[seq_idx]} finished training. Best epoch: {trainer.best_epoch + 1} "
                            f"Best val accuracy: {trainer.best_val_acc} Best val loss: {trainer.best_val_loss}")
                        print('\n')
                record_dict[sequence_name] = record_list
            records.append(record_dict)

        return records
