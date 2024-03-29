import sys

sys.path.append('/u/erdos/students/xcui32/SequentialTraining')
sys.path.append('/u/erdos/students/xcui32')
sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages')
sys.path.append('/u/erdos/students/xcui32/.local/lib/python3.6/site-packages')

import argparse
import os
from typing import Optional, List
import numpy as np
from utils import metrics
import torch
from SequentialTraining.regimens import get_regimen_dataloaders
from SequentialTraining.helpers import get_dataset
from config import none_or_str, Config
from trainer import Trainer
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import torchvision
from SequentialTraining.utils.coco_utils import get_big_coco_classes
from SequentialTraining.performance import get_avg_std
from tabulate import tabulate

pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description='Main module to run')
parser.add_argument('--regimens', type=none_or_str, nargs='?', const=None,
                    help="a list of regimens, separated by comma; "
                         "available options are: 'stb_endsame', 'bts_startsame', 'llo', 'random_oneseq', 'random1', 'single'")

# dataset
parser.add_argument('--dataset_name', default=None, nargs='?', type=none_or_str, const=None,
                    help='dataset name, from one of: VOC, Imagenet, CIFAR10')
parser.add_argument('--train_annotation_path', default=None, nargs='?', type=none_or_str, const=None,
                    help='a list of train annotation root path, separated by comma')
parser.add_argument('--train_image_path', default=None, type=none_or_str, nargs='?', const=None,
                    help='a list of train images root path, separated by comma')
parser.add_argument('--test_annotation_path', default=None, nargs='?', const=None, type=none_or_str,
                    help='test annotation root path')
parser.add_argument('--test_image_path', type=none_or_str, nargs='?', const=None, help='test images root path')

# coco specific

parser.add_argument('--min_image_per_class', type=int, nargs='?', default=None, const=None,
                    help='minimum number of valid images required for COCO per class')
parser.add_argument('--max_image_per_class', type=int, nargs='?', default=None, const=None,
                    help='max number of valid images for COCO per class')

parser.add_argument('--num_samples_to_use', type=int, nargs='?', const=None, default=None,
                    help='# of images to use for training')
parser.add_argument('--test_size', type=float, nargs='?', const=0.3, default=0.3,
                    help='Ratio for how many data to split from the given data for test set'
                         'Only for Imagenet and COCO')
parser.add_argument('--num_classes', type=int, nargs='?', const=None, default=None, help='# of classes to use')
parser.add_argument('--cls_to_use', type=none_or_str, nargs='?', const=None, default=None,
                    help='# of images to use for training')

# image args
default_sizes = '0.2, 0.4, 0.6, 0.8, 1'
parser.add_argument('--input_size', default=150, type=int, nargs='?',
                    help='width/height of input image size. Default 150 -- (150, 150)')
parser.add_argument('--sizes', default=default_sizes, type=str, const=default_sizes, nargs='?',
                    help='sizes group for each regimens, separated by comma')
parser.add_argument('--resize_method', default='long', type=str, const='long', nargs='?',
                    help='Method to resize images. Either "long": resize with longer side of the image to input_size * sizes and add padding or '
                         '"adjust": resize to input_size * input_size without padding')
# training args
parser.add_argument('--lr', default=0.1, const=0.1, nargs='?', type=float, help='learning rate')
parser.add_argument('--epoch_schedule', default=None, const=None, nargs='?', type=str, help='a list of epochs for each training size group')
parser.add_argument('--epoch', default=200, const=200, nargs='?', type=int, help='num of epochs')
parser.add_argument('--model', type=str, help='model name from pytorch')
parser.add_argument('--num_workers', default=16, const=16, type=int, nargs='?', help='number of workers for dataloader')
parser.add_argument('--batch_size', default=128, const=128, type=int, nargs='?', help='batch size')
parser.add_argument('--max_norm', default=None, const=None, nargs='?', type=int, help='max norm for gradient clipping')
parser.add_argument('--reset_lr', default=None, const=None, nargs='?', type=float,
                    help='if not None, lr will be reset to =reset_lr at the start of each size group')
parser.add_argument('--early_stop_patience', default=20, const=20, nargs='?', type=int,
                    help='number of patience to wait before early stop')
parser.add_argument('--lr_patience', default=3, const=3, nargs='?', type=int,
                    help='number of epochs to wait before decreasing lr')
parser.add_argument('--min_lr', default=0.00001, const=0.00001, nargs='?', type=float, help='minimum learning rate')
parser.add_argument('--n_folds', default=5, const=5, nargs='?', type=int, help='number of folds')
parser.add_argument('--n_folds_to_use', default=5, const=5, nargs='?', type=int, help='number of folds to use')
parser.add_argument('--device', default=None, const=None, nargs='?', type=str, help='number of folds to use')

# result
parser.add_argument('--save_progress_ckpt', type=str, default=False, const=False, nargs='?',
                    help='whether to save all checkpoints during training process')
parser.add_argument('--save_result_ckpt', type=str, default=False, const=False, nargs='?',
                    help='whether to save all checkpoints for each regimen')
parser.add_argument('--result_path', type=str, help='path to result directory')
args = parser.parse_args()

# some default configurations
dataset_VOC = {
    'train_annotation_path': [
        '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/train/annotations',
        '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/val/annotations'],
    'train_image_path': ['/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/train/root',
                         '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/val/root'],
    'test_annotation_path': [
        '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/test/annotations'],
    'test_image_path': ['/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/test/root']}

dataset_CIFAR10 = {'train_annotation_path': None,
                   'train_image_path': ['/u/erdos/students/xcui32/SequentialTraining/datasets/cifar10/train'],
                   'test_annotation_path': None,
                   'test_image_path': ['/u/erdos/students/xcui32/SequentialTraining/datasets/cifar10/test']}

dataset_Imagenet = {'train_annotation_path': ['/u/erdos/cnslab/imagenet-bndbox/bndbox'],
                    'train_image_path': ['/u/erdos/cnslab/imagenet-distinct'],
                    'test_annotation_path': ['/u/erdos/cnslab/imagenet-bndbox/bndbox'],
                    'test_image_path': ['/u/erdos/cnslab/imagenet-distinct']}

dataset_COCO = {'train_annotation_path': ['/u/erdos/cnslab/coco/annotations/classification_train2017.json'],
                'train_image_path': ['/u/erdos/cnslab/coco/train'],
                'test_annotation_path': ['/u/erdos/cnslab/coco/annotations/classification_test2017.json'],
                'test_image_path': ['/u/erdos/cnslab/coco/test']}

if args.dataset_name == 'VOC':
    dataset_paths = dataset_VOC
elif args.dataset_name == 'Imagenet':
    dataset_paths = dataset_Imagenet
elif args.dataset_name == 'CIFAR10':
    dataset_paths = dataset_CIFAR10
elif args.dataset_name == 'COCO':
    dataset_paths = dataset_COCO
else:
    raise Exception(f"{args.dataset_name} has not been implemented")

all_regimens = ['stb_endsame', 'bts_startsame', 'random-single', 'random_1group', 'random', 'single']

optimizer_kwargs = {'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.0001}
scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': args.lr_patience, 'min_lr': args.min_lr}
device = "cuda:0" if torch.cuda.is_available() else "cpu"

if 'COCO' in args.dataset_name:
    cls_to_use = args.cls_to_use if args.cls_to_use else \
        get_big_coco_classes(args.input_size, path_to_json=args.train_annotation_path if args.train_annotation_path
        else dataset_COCO['train_annotation_path'][0],
        min_image_per_class=args.min_image_per_class, num_classes=args.num_classes)
else:
    cls_to_use = args.cls_to_use

if args.epoch_schedule:
    epoch_schedule = eval(args.epoch_schedule)
else:
    epoch_schedule = None

config = Config(regimens=all_regimens if (args.regimens is None) or (args.regimens == 'all') else [x.strip() for x in
                                                                                                   args.regimens.split(
                                                                                                       ',')],
                dataset_name=args.dataset_name,
                train_image_path=dataset_paths['train_image_path'],
                train_annotation_path=dataset_paths['train_annotation_path'],
                test_annotation_path=dataset_paths['test_annotation_path'],
                test_image_path=dataset_paths['test_image_path'],
                num_samples_to_use=args.num_samples_to_use, test_size=args.test_size,
                num_classes=args.num_classes, cls_to_use=cls_to_use,
                sizes=[eval(x.strip()) for x in args.sizes.split(',')],
                input_size=(args.input_size, args.input_size),
                resize_method=args.resize_method,
                model=args.model, epoch_schedule=epoch_schedule,
                epoch=args.epoch, min_lr=args.min_lr,
                n_folds=args.n_folds, n_folds_to_use=args.n_folds_to_use,
                early_stop_patience=args.early_stop_patience, max_norm=args.max_norm,
                reset_lr=args.reset_lr,
                optim_kwargs=optimizer_kwargs, scheduler_kwargs=scheduler_kwargs,
                device=args.device if args.device else device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                result_path=args.result_path,
                save_progress_ckpt=eval(args.save_progress_ckpt) if args.save_progress_ckpt else False,
                save_result_ckpt=eval(args.save_result_ckpt) if args.save_result_ckpt else False)

optimizer_object = torch.optim.SGD
criterion_object = torch.nn.CrossEntropyLoss
scheduler_object = torch.optim.lr_scheduler.ReduceLROnPlateau


def train_regimen(regimen_name: str, train_indices: Optional[List[int]] = None, test_indices: Optional[List[int]] = None):
    regimen = get_regimen_dataloaders(input_size=config.input_size, sizes=config.sizes, regimen=regimen_name,
                                      num_samples_to_use=config.num_samples_to_use,
                                      dataset_name=config.dataset_name, image_roots=config.train_image_path,
                                      annotation_roots=config.train_annotation_path,
                                      min_image_per_class=config.min_image_per_class,
                                      cls_to_use=config.cls_to_use, num_classes=config.num_classes,
                                      train_indices=train_indices)

    # get all test dataloaders
    test_dataloaders = []
    for size in sorted(config.sizes):
        test_dataset = get_dataset(dataset_name=config.dataset_name, size=config.input_size, p=size,
                                   image_roots=config.test_image_path,
                                   indices=test_indices, annotation_roots=config.test_annotation_path,
                                   resize_method=config.resize_method, train=False,
                                   cls_to_use=config.cls_to_use, num_classes=config.num_classes)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True,
                                     num_workers=config.num_workers, pin_memory=True)
        test_dataloaders.append(test_dataloader)

    criterion = criterion_object()
    records = []

    test_result = []

    best_epoch = {}

    for fold, sequence in enumerate(regimen):

        model_best_states = {}
        record_dict = {}
        test_df = pd.DataFrame(index=sorted(config.sizes) + ['origin'])
        if config.n_folds_to_use:
            if fold >= config.n_folds_to_use:
                break
        print(f'Fold: {fold}')
        for sequence_name, sequence_dataloaders in sequence:
            # e.g. sequence_name: "[0.8, 1]"
            # sequence_dataloaders: [(train_dataloader_0.8, val_dataloader_0.8), (train_dataloader_1, val_dataloader_1)]
            print(f'==> Training {config.model} on {sequence_name}')
            if config.model == 'inception_v3' or config.model == 'googlenet':
                model = eval(
                    'torchvision.models.' + config.model + f'(num_classes={config.num_classes}, aux_logits=False)')
            else:
                model = eval('torchvision.models.' + config.model + f'(num_classes={config.num_classes})')
            model = model.to(config.device)
            epochs_per_size = int(np.ceil(config.epoch / len(sequence_dataloaders)))
            sequence_list = eval(sequence_name)  # [0.6, 0.8, 1]

            record_list = []
            best_epoch_list = []

            for seq_idx, (train_dataloader, val_dataloader) in enumerate(sequence_dataloaders):
                print(
                    f'# of trains: {len(train_dataloader) * config.batch_size} # of vals: {len(val_dataloader) * config.batch_size}')
                if str(sequence_list[:seq_idx + 1]) in model_best_states:
                    model.load_state_dict(model_best_states[str(sequence_list[:seq_idx + 1])])
                    record_list = record_list + record_dict[str(sequence_list[:seq_idx + 1])]
                    print(f"Sequence {sequence_list[:seq_idx + 1]} already in state dictionary, jumped")

                else:

                    optimizer = optimizer_object(model.parameters(), **config.optim_kwargs)
                    if config.reset_lr and seq_idx > 0:
                        for g in optimizer.param_groups:
                            g['lr'] = config.reset_lr
                    if scheduler_object:
                        scheduler = scheduler_object(optimizer, **config.scheduler_kwargs)
                    if str(sequence_list[:seq_idx]) in model_best_states:
                        model.load_state_dict(model_best_states[str(sequence_list[:seq_idx])])
                        print(f'Loaded best state dict for {str(sequence_list[:seq_idx])}')

                    elif seq_idx == 0:
                        pass
                    else:
                        raise ValueError(
                            "Uh Something went wrong... the previous sequence is not in the state "
                            "dict...")

                    trainer = Trainer(criterion=criterion, patience=config.early_stop_patience, device=config.device)
                    if regimen_name.startswith(('bts', 'stb', 'as_is', 'random-single')) and config.epoch_schedule:
                        num_epoch = config.epoch_schedule[seq_idx]
                    else:
                        num_epoch = epochs_per_size

                    print(f'==> Current group: {sequence_list[seq_idx]} num_epoch: {num_epoch}')
                    for epoch in range(num_epoch):

                        train_loss, train_acc, val_loss, val_acc, lr = trainer.train(epoch, model, train_dataloader,
                                                                                     val_dataloader, optimizer, config.max_norm)

                        print(
                            'Epoch [{}/{}] Training Loss: {:.4f} Training Acc: {:.3f} Val Loss: {:.4f} Val Acc: {:.3f} lr: {:.4f}'
                                .format(epoch + 1, epochs_per_size, train_loss, train_acc, val_loss, val_acc, lr))

                        if trainer.is_loss_lower:
                            model_best_states[str(sequence_list[:seq_idx + 1])] = model.state_dict()
                            if config.save_progress_ckpt:
                                fold_dir = os.path.join(config.result_path, 'checkpoints', f'fold_{fold}')
                                if not os.path.isdir(fold_dir):
                                    os.mkdir(fold_dir)
                                torch.save(model.state_dict(),
                                           os.path.join(fold_dir, f'{str(sequence_list[:seq_idx + 1])}.pt'))

                        else:
                            if trainer.stop:
                                print(" --- Early Stopped ---")
                                break
                        if scheduler_object:
                            scheduler.step(val_loss)

                    record_list.append(trainer)
                    best_epoch_list.append(trainer.best_epoch + 1)
                    model.load_state_dict(model_best_states[str(sequence_list[:seq_idx + 1])])
                    print(
                        f"Group: {sequence_list[seq_idx]} finished training. Best epoch: {trainer.best_epoch + 1} "
                        f"Best val accuracy: {trainer.best_val_acc} Best val loss: {trainer.best_val_loss}")
                    print('\n')
            record_dict[sequence_name] = record_list
            if sequence_name not in best_epoch:
                best_epoch[sequence_name] = []
            else:
                best_epoch[sequence_name].append(best_epoch_list)

            # testing

            # append the original image with the last size in the sequence

            if config.dataset_name != 'COCO':
                original_testset = get_dataset(dataset_name=config.dataset_name, size=config.input_size, p=float(sequence_list[-1]),
                                               image_roots=config.test_image_path,
                                               indices=test_indices, annotation_roots=config.test_annotation_path,
                                               resize_method=config.resize_method, train=False,
                                               cls_to_use=config.cls_to_use, num_classes=config.num_classes,
                                               origin=True)
                original_dataloader = DataLoader(original_testset, batch_size=config.batch_size, shuffle=True,
                                                 num_workers=config.num_workers, pin_memory=True)
                test_dataloaders_with_origin = test_dataloaders + [original_dataloader]
            else:
                test_dataloaders_with_origin = test_dataloaders

            model.eval()
            test_list = []
            test_target = set()
            for test_dataloader in test_dataloaders_with_origin:
                acc_1 = 0
                with torch.no_grad():
                    for images, labels in test_dataloader:
                        images = images.to(config.device)
                        labels = labels.to(config.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)

                        acc = metrics.accuracy(outputs, labels, (1,))
                        acc_1 += acc[0]
                        test_target.update(list(labels.cpu().numpy()))

                    acc_1 = acc_1 / len(test_dataloader)
                    acc_1 = float(acc_1.cpu().numpy()[0])

                test_list.append(acc_1)
            test_df[sequence_name] = test_list

        test_result.append(test_df)
        records.append(record_dict)

    print("Test results:")
    avg, std = get_avg_std(test_result)
    print('Accuracy: ')
    print(tabulate(avg, headers='keys', tablefmt='psql'))
    print('std: ')
    print(tabulate(std, headers='keys', tablefmt='psql'))
    print('Average best epochs: ')
    for k,v in best_epoch.items():
        print(k, np.mean(v, axis=0))

    result = {'config': config,
              'train': records,
              'test': test_result}
    with open(os.path.join(config.result_path, regimen_name), 'wb') as handle:
        pickle.dump(result, handle)


def main():
    if not os.path.isdir(config.result_path):
        os.mkdir(config.result_path)

    train_dataset, dataset_classes = get_dataset(dataset_name=config.dataset_name, size=config.input_size, p=1,
                                                 image_roots=config.train_image_path, cls_to_use=config.cls_to_use,
                                                 num_classes=config.num_classes,
                                                 annotation_roots=config.train_annotation_path, return_classes=True)
    num_samples = len(train_dataset)
    config.num_classes = len(dataset_classes)
    print('Number of total train (before train-val split) samples: ', num_samples)
    print('Number of classes: ', config.num_classes)

    for name, value in config:
        print(f'{name}: {value}')

    # for datasets that don't have separate train/test
    if 'Imagenet' in config.dataset_name:

        np.random.seed(config.random_seed)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        train_indices = indices[:int((1 - config.test_size) * len(indices))]
        test_indices = indices[int((1 - config.test_size) * len(indices)):]

        for regimen in config.regimens:
            print(f'==>Training {regimen}\n')
            train_regimen(regimen_name=regimen, train_indices=train_indices, test_indices=test_indices)
            print('\n')

    else:
        for regimen in config.regimens:
            print(f'==>Training {regimen}\n')
            train_regimen(regimen_name=regimen)
            print('\n')


if __name__ == '__main__':
    main()
