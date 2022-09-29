import os
from collections import Counter
from pathlib import Path

import torch
import torchvision
from typing import Tuple, Callable, List
from utils.data_utils import GenerateBackground, ResizeImageLoader
from torch_datasets import VOCImageFolder
import numpy as np
from torch.utils.data import DataLoader
import json


class RunModel:
    def __init__(self,
                 train_root_path: str,
                 val_root_path: str,
                 test_root_path: str,
                 dataset_name: str,
                 target_distances: List[float],
                 training_mode: str = 'stb',
                 n_distances: int = 5,
                 background: Callable = GenerateBackground(bg_type='fft'),
                 size: Tuple[int, int] = (150, 150),
                 cls_to_use: List[str] = None,
                 model_name: str = 'alexnet',
                 epochs: int = 200,
                 batch_size: int = 128,
                 val_size: float = 1,
                 num_workers: int = 4,
                 device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
                 random_seed: int = 40,
                 result_dirpath: str = Path(__file__).parent / "datasets/results",
                 save_checkpoints: bool = True,
                 verbose: int = 1):
        """

        :param train_root_path: root path for training data.
        :param test_root_path: root path for testing data
        :param dataset_name: name of the dataset.
            VOC:
                Expect train_root_path folder structure:
                    train_root_path/
                    |-- root/
                    |   |-- class1
                    |   |   |-- img1.jpg
                    |   |-- class2
                    |   |   |-- img2.jpg
                    |-- Annotation/
                    |   |-- class1
                    |   |   |-- img1.xml
                    |   |-- class2
                    |   |   |-- img2.xml
        :param target_distances:
                                list: a list of target distances, ordered by training_mode
                                float: single target distances
        :param training_mode: for each distance specified in the target_distances list:
                                stb:
                                bts:
                                llo (leave last out):
                                random: all random, specified by target_distances
                                single: single distances -- equivalent to n_distances == 0
        :param n_distances: if not None/greater than 0, will use n different ratios before the target ratios, order specified by training_mode
                            if 0: single distance
                            if None: use distance ratios specified in target_distances
        :param background: callable
        :param result_dirpath: path to directory for saving model_name results. If None, model_name will not be saved
        """
        self.training_root_path = train_root_path
        self.val_root_path = val_root_path
        self.test_root_path = test_root_path
        self.dataset_name = dataset_name
        self.target_distances = target_distances
        self.training_mode = training_mode
        self.n_distances = n_distances
        self.background = background
        self.size = size
        self.cls_to_use = cls_to_use
        self.batch_size = batch_size
        self.val_size = val_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.model_name = model_name
        self.device = device
        self.random_seed = random_seed
        self.result_dirpath = result_dirpath
        self.save_checkpoints = save_checkpoints
        self.verbose = verbose

        print(' ------ Pipeline with following parameters ------')
        for key, value in {k: v for k, v in self.__dict__.items() if not k.startswith('__')}.items():
            print(key, ": ", value)
        self.train_datasets = {}
        self.test_datasets = {}
        self.val_datasets = {}
        self.num_classes = None if cls_to_use is None else len(cls_to_use)
        self.all_training_loss = {}
        self.all_val_loss = {}
        self.all_val_acc_top1 = {}
        self.test_acc_top1 = {}
        self.models_statedict = []

    def load_datasets(self):
        test_annotation_root_path = os.path.join(self.test_root_path, 'annotations')
        train_annotation_root_path = os.path.join(self.training_root_path, 'annotations')
        val_annotation_root_path = os.path.join(self.val_root_path, 'annotations')
        train_image_root_path = os.path.join(self.training_root_path, 'root')
        val_image_root_path = os.path.join(self.val_root_path, 'root')
        test_image_root_path = os.path.join(self.test_root_path, 'root')

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        if self.dataset_name == 'VOC':

            val_datasets_to_combine = []
            train_datasets_to_combine = []
            for target_distances in self.target_distances:

                # save testing dataset for each distance
                test_loader = ResizeImageLoader(self.size, p=target_distances,
                                                background_generator=self.background,
                                                annotation_root_path=test_annotation_root_path)
                test_dataset = VOCImageFolder(cls_to_use=self.cls_to_use, root=test_image_root_path,
                                              transform=transform,
                                              loader=test_loader)
                # save num_of_classes, if it's None
                if self.num_classes is None:
                    self.num_classes = len(test_dataset.classes)

                self.test_datasets[str(target_distances)] = test_dataset

                # For validation set, use the mixed distances

                # TODO: consider using separate validation set for each distance?
                val_loader = ResizeImageLoader(self.size, p=target_distances,
                                               background_generator=self.background,
                                               annotation_root_path=val_annotation_root_path)
                val_dataset = VOCImageFolder(cls_to_use=self.cls_to_use, root=val_image_root_path,
                                             transform=transform, loader=val_loader)

                val_datasets_to_combine.append(val_dataset)


                # save training dataset objects for each distance
                train_loader = ResizeImageLoader(self.size, p=target_distances,
                                                 background_generator=self.background,
                                                 annotation_root_path=train_annotation_root_path)
                train_dataset = VOCImageFolder(cls_to_use=self.cls_to_use, root=train_image_root_path,
                                               transform=transform, loader=train_loader)
                if self.training_mode == 'llo' or self.training_mode == 'random':
                    train_datasets_to_combine.append(train_dataset)
                else:
                    self.train_datasets[str(target_distances)] = train_dataset

            combined_datasets = torch.utils.data.ConcatDataset(val_datasets_to_combine)
            indices = np.arange(len(combined_datasets))
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
            val_dataset_same_for_all = torch.utils.data.Subset(combined_datasets,
                                                               indices[:len(val_datasets_to_combine[0])])

            # use the same val set for all distances
            for target_distances in self.target_distances:
                self.val_datasets[str(target_distances)] = val_dataset_same_for_all

            if self.training_mode == 'llo':
                for i in range(len(self.target_distances)):
                    random_distances = self.target_distances[:i] + self.target_distances[i + 1:]
                    datasets = []
                    for random_distance in random_distances:

                        datasets.append(self.train_datasets[str(random_distance)])

                    combined_datasets = torch.utils.data.ConcatDataset(datasets)
                    indices = np.arange(len(combined_datasets))
                    np.random.seed(self.random_seed)
                    np.random.shuffle(indices)
                    indices_dataset = np.array_split(indices, len(self.target_distances) - 1)

                    for j, idx in enumerate(indices_dataset):
                        self.train_datasets[f'llo_{self.target_distances[i]}_random{j}'] = torch.utils.data.Subset(combined_datasets, idx)

            if self.training_mode == 'random':
                datasets = []
                for train_distance in self.target_distances:

                    datasets.append(self.train_datasets[str(train_distance)])

                combined_datasets = torch.utils.data.ConcatDataset(datasets)
                indices = np.arange(len(combined_datasets))
                np.random.seed(self.random_seed)
                np.random.shuffle(indices)
                indices_dataset = np.array_split(indices, len(self.target_distances))

                for j, idx in enumerate(indices_dataset):
                    self.train_datasets[f'random{j}'] = torch.utils.data.Subset(combined_datasets, idx)



    def run(self,
            criterion_object: Callable,
            optimizer_object: Callable,
            n_folds: int = None,
            early_stopping: bool = True,
            patience: int = 2,
            **kwargs) -> None:
        """

        :param patience: tolerance for loss increase before early stopping
        :param criterion_object:
        :param optimizer_object:
        :param early_stopping:
        :param n_folds:
        :param kwargs: only for optimizer
        :return:
        """
        assert self.train_datasets is not None and self.test_datasets is not None and self.num_classes is not None, \
            "Datasets is None. Please run RunModel.load_datasets() first "

        val_dataloader = DataLoader(self.val_datasets[0], batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers)
        best_state_dict = {}

        sequence_best_state = {}




        for name, sequence in self.train_datasets:
            print(f"----- Training {self.model_name} with sequence: {name} -----")
            model = eval('models.' + self.model_name + f'(num_classes={self.num_classes}, pretrained={False})')
            model = model.to(self.device)
            criterion = criterion_object()
            optimizer = optimizer_object(model.parameters(), **kwargs)

            self.all_training_loss[str(name)] = []
            self.all_val_loss[str(name)] = []
            self.all_val_acc_top1[str(name)] = []
            epochs_per_distance = int(np.ceil(self.epochs / len(sequence)))

            best_sub_model, best_sub_optimizer = None, None

            for seq_idx, (distance, dataset) in enumerate(sequence):

                if best_sub_model is not None:
                    model.load_state_dict(best_sub_model)
                if best_sub_optimizer is not None:
                    optimizer.load_state_dict(best_sub_optimizer)
                train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers)

                sub_training_loss, sub_val_loss = [], []
                val_top1_acc = []

                # current group's best losses
                patience_count = 0
                best_epoch = 0
                best_val_acc = 0
                best_val_loss = np.inf
                # TODO: validation set for separate distances?
                print(f'Current group: {distance}')
                for epoch in range(epochs_per_distance):

                    # training
                    model.train()

                    training_loss_per_pass = 0
                    for i, (images, labels) in enumerate(train_loader):
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        training_loss_per_pass += loss.item()
                    sub_training_loss.append(training_loss_per_pass)

                    # validation
                    correct = 0
                    total = 0
                    val_loss_per_pass = 0
                    model.eval()
                    predicted_all = []
                    with torch.no_grad():
                        for images, labels in val_dataloader:
                            images = images.to(self.device)
                            labels = labels.to(self.device)
                            outputs = model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            val_loss = criterion(outputs, labels)
                            val_loss_per_pass += val_loss.item()
                            predicted_all.extend(predicted.cpu().detach().tolist())
                    acc = correct / total
                    val_top1_acc.append(acc)
                    sub_val_loss.append(val_loss_per_pass)
                    if self.verbose >= 1:
                        predicted_count = sorted(Counter(predicted_all).items())
                        print(f"predicted label frequency: {predicted_count}")

                    print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Top1 '
                          'Accuracy: {:.4f}'
                          .format(epoch + 1, epochs_per_distance, training_loss_per_pass, val_loss_per_pass, acc))
                    if val_loss_per_pass <= best_val_loss:
                        best_val_loss = val_loss_per_pass
                        best_sub_model = model.state_dict()
                        best_sub_optimizer = optimizer.state_dict()

                        best_state_dict[str(sequence[:seq_idx + 1])] = (model.state_dict(), optimizer.state_dict())

                        patience_count = 0
                        best_epoch = epoch
                        best_val_acc = acc
                    else:
                        patience_count += 1
                        if early_stopping and patience_count >= patience:
                            print(" --- Early Stopped ---")
                            break
                print(
                    f"Patch distance: {distance} finished training. Best epoch: {best_epoch} Best val accuracy: {best_val_acc} Best val loss: {best_val_loss}")
                print('\n')
                self.all_training_loss[str(name)].append((str(distance), sub_training_loss))
                self.all_val_loss[str(name)].append((str(distance), sub_val_loss))
                self.all_val_acc_top1[str(name)].append((str(distance), val_top1_acc))
            self.models_statedict.append((name, model.state_dict()))
        print('-' * 20, 'All training done', '-' * 20)

    def evaluate(self, suffix: str = ''):
        test_dataloaders = [
            (name, DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers))
            for name, dataset in self.test_datasets]

        if not os.path.isdir(self.result_dirpath):
            os.mkdir(self.result_dirpath)
        sub_dir = os.path.join(self.result_dirpath,
                               f"{self.dataset_name}_{self.model_name}_{self.training_mode}_{self.background}_{len(self.cls_to_use)}class_{suffix}")
        os.mkdir(sub_dir)

        for name, model_params in self.models_statedict:
            print(f'---- Testing model trained on sequence: {name} ----')
            if self.save_checkpoints:
                torch.save(model_params, os.path.join(sub_dir, '-'.join([str(i) for i in name])))

            self.test_acc_top1[name] = []
            model = eval('torchvision.models.' + self.model_name)(pretrained=False)
            if self.model_name == 'alexnet':
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(num_ftrs, self.num_classes)
            model.load_state_dict(model_params)
            model.to(self.device)
            model.eval()

            for distance, test_dataloader in test_dataloaders:

                correct = 0
                total = 0
                acc = -1
                with torch.no_grad():
                    for images, labels in test_dataloader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        acc = correct / total
                    self.test_acc_top1[name].append((distance, acc))
                    print(f"Test set distance: {distance} Top 1 Accuracy: {acc}")

        result = {
            'all_training_loss': self.all_training_loss,
            'all_val_loss': self.all_val_loss,
            'all_val_acc_top1': self.all_val_acc_top1,
            'test_acc_top1': self.test_acc_top1}
        with open(os.path.join(sub_dir, 'acc_n_loss.json'), 'w') as f:
            json.dump(result, f)
