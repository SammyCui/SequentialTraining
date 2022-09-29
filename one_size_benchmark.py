import os
from pathlib import Path

import torch
import torchvision
from typing import Tuple, Callable, List
from utils.data_utils import GenerateBackground, ResizeImageLoader
from torch_datasets import VOCImageFolder
import numpy as np
from torch.utils.data import DataLoader
import json


class VOCSingleTest:
    def __init__(self,
                 train_root_path: str,
                 val_root_path: str,
                 test_root_path: str,
                 target_size: float,
                 background: Callable = GenerateBackground(bg_type='fft'),
                 size: Tuple[int, int] = (150, 150),
                 cls_to_use: List[str] = None,
                 model_name: str = 'alexnet',
                 epochs: int = 200,
                 resize_method: str = 'long',
                 batch_size: int = 128,
                 val_size: float = 1,
                 num_workers: int = 4,
                 n_folds: int = None,
                 device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
                 random_seed: int = 40,
                 result_dirpath: str = Path(__file__).parent / "datasets/results",
                 save_checkpoints: bool = True,
                 verbose: int = 1
                 ):

        self.training_root_path = train_root_path
        self.val_root_path = val_root_path
        self.test_root_path = test_root_path
        self.target_size = target_size
        self.background = background
        self.size = size
        self.cls_to_use = cls_to_use
        self.batch_size = batch_size
        self.val_size = val_size
        self.epochs = epochs
        self.resize_method = resize_method
        self.n_folds = n_folds
        self.num_workers = num_workers
        self.model_name = model_name
        self.device = device
        self.random_seed = random_seed
        self.result_dirpath = result_dirpath
        self.save_checkpoints = save_checkpoints
        self.verbose = verbose

        print(' ------ parameters ------')
        for key, value in {k: v for k, v in self.__dict__.items() if not k.startswith('__')}.items():
            print(key, ": ", value)


    def run_test(self,
                 criterion_object: Callable,
                 optimizer_object: Callable,
                 scheduler_object: Callable = None,
                 early_stopping: bool = True,
                 patience: int = 20,
                 max_norm: int = None,
                 optim_kwargs: dict = None,
                 scheduler_kwargs: dict = None
                 ):
        test_annotation_root_path = os.path.join(self.test_root_path, 'annotations')
        train_annotation_root_path = os.path.join(self.training_root_path, 'annotations')
        val_annotation_root_path = os.path.join(self.val_root_path, 'annotations')
        train_image_root_path = os.path.join(self.training_root_path, 'root')
        val_image_root_path = os.path.join(self.val_root_path, 'root')
        test_image_root_path = os.path.join(self.test_root_path, 'root')

        training_len = len(VOCImageFolder(cls_to_use=self.cls_to_use, root=train_image_root_path))
        # testing_len = len(VOCDataset(cls_to_use=self.cls_to_use, root=test_image_root_path))
        val_len = len(VOCImageFolder(cls_to_use=self.cls_to_use, root=val_image_root_path))

        val_size = 1 / self.n_folds
        num_train = training_len + val_len
        cv_indices = list(range(num_train))
        np.random.seed(self.random_seed)
        np.random.shuffle(cv_indices)
        split = int(np.floor(val_size * num_train))

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_loader = ResizeImageLoader(self.size, p=self.target_size,
                                        background_generator=self.background,
                                        annotation_root_path=test_annotation_root_path)
        test_dataset = VOCImageFolder(cls_to_use=self.cls_to_use, root=test_image_root_path,
                                      transform=transform,
                                      loader=test_loader)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        num_classes = len(test_dataset.classes)

        train_loss_dict = {}
        val_loss_dict = {}
        test_acc_dict = {}

        for fold in range(self.n_folds):


            split1 = int(np.floor(fold * split))
            split2 = int(np.floor((fold + 1) * split))
            val_idx = cv_indices[split1:split2]
            train_idx = np.append(cv_indices[:split1], cv_indices[split2:])
            train_idx = train_idx.astype('int32')
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = ResizeImageLoader(self.size, p=self.target_size,
                                             background_generator=self.background,
                                             annotation_root_path=train_annotation_root_path)
            train_dataset = VOCImageFolder(cls_to_use=self.cls_to_use, root=train_image_root_path,
                                           transform=transform, loader=train_loader)
            val_loader = ResizeImageLoader(self.size, p=self.target_size,
                                           background_generator=self.background,
                                           annotation_root_path=val_annotation_root_path)
            val_dataset = VOCImageFolder(cls_to_use=self.cls_to_use, root=val_image_root_path,
                                         transform=transform, loader=val_loader)

            dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

            train_dataloader = torch.utils.data.DataLoader(
                dataset, sampler=train_sampler, batch_size=self.batch_size,
                num_workers=self.num_workers, pin_memory=True)

            model = eval('models.' + self.model_name + f'(num_classes={num_classes}, pretrained={False})')
            model = model.to(self.device)
            criterion = criterion_object()
            optimizer = optimizer_object(model.parameters(), **optim_kwargs)
            if scheduler_object:
                scheduler = scheduler_object(optimizer, **scheduler_kwargs)

            training_loss = []
            val_acc, val_loss = [], []
            best_val_loss = np.inf
            patience_count = 0
            for epoch in range(self.epochs):

                # training
                model.train()

                training_loss_per_pass = 0
                for i, (images, labels) in enumerate(train_dataloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()

                    if max_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    optimizer.step()
                    training_loss_per_pass += loss.item()
                training_loss.append(training_loss_per_pass)

                # validation

                correct = 0
                total = 0
                val_loss_per_pass = 0
                model.eval()
                val_batch_size = min(len(val_dataset), self.batch_size)
                val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size,
                                            sampler=val_sampler,
                                            num_workers=self.num_workers, pin_memory=True)
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
                acc = correct / total

                val_acc.append(acc)
                val_loss.append(val_loss_per_pass)

                print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation '
                      'Accuracy: {:.4f}'
                      .format(epoch + 1, self.epochs, training_loss_per_pass,
                              val_loss_per_pass, acc))

                if val_loss_per_pass <= best_val_loss:
                    best_val_loss = val_loss_per_pass
                    best_state_dict = model.state_dict()
                    if self.save_checkpoints:
                        torch.save(best_state_dict, os.path.join(self.result_dirpath, f'{self.model_name}_{fold}.pt'))

                    patience_count = 0

                else:
                    patience_count += 1
                    if early_stopping and patience_count >= patience:
                        print(" --- Early Stopped ---")
                        break

                if scheduler_object:
                    scheduler.step(val_loss_per_pass)

            train_loss_dict[fold] = training_loss
            val_loss_dict[fold] = val_loss

            # Test
            print("---------- Testing -----------")
            best_model = eval('models.' + self.model_name + f'(num_classes={num_classes}, pretrained={False})')
            best_model = best_model.to(self.device)
            best_model.load_state_dict(best_state_dict)
            best_model.to(self.device)
            best_model.eval()


            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = best_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = correct / total

            test_acc_dict[fold] = acc

            print(f"Fold: {fold} Top 1 Accuracy: {acc}")

        with open(os.path.join(self.result_dirpath, 'acc_n_loss.json'), 'w') as f:
            json.dump({'training_loss': train_loss_dict,
                       'val_loss': val_loss_dict,
                       'test_acc': test_acc_dict}, f)

if __name__ == '__main__':
    cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor']
    train_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/train"
    val_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/val"
    test_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/test"
    result_dirpath = Path(__file__).parent / "results/VOC8AlexnetFFTTest_1"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    pipeline_params = {'train_root_path': train_root_path,
                       'val_root_path': val_root_path,
                       'test_root_path': test_root_path,
                       'target_size': 1,
                       'background': GenerateBackground(bg_type='fft', bg_color=(0, 0, 0)),
                       'size': (150, 150),
                       'device': device,
                       'cls_to_use': cats,
                       'verbose': 0,
                       'resize_method': 'long',
                       'epochs': 200,
                       'n_folds': 5,
                       'batch_size': 128,
                       'num_workers': 8,
                       'model_name': 'alexnet',
                       'result_dirpath': result_dirpath,
                       'random_seed': 40,
                       'save_checkpoints': False}

    optimizer_kwargs = {'lr': 0.001, 'momentum': 0.9}
    scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5}
    fit_params = {'criterion_object': torch.nn.CrossEntropyLoss,
                  'optimizer_object': torch.optim.SGD,
                  'scheduler_object': None, #torch.optim.lr_scheduler.ReduceLROnPlateau,
                  'patience': 20,
                  'optim_kwargs': optimizer_kwargs,
                  'scheduler_kwargs': scheduler_kwargs}
    VOCpipeline = VOCSingleTest(**pipeline_params)
    VOCpipeline.run_test(**fit_params)