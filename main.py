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
    'test_annotation_path': ['/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/test/annotations'],
    'test_image_path': ['/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/test/root']}

dataset_CIFAR10 = {'train_annotation_path': None,
                   'train_image_path': ['/u/erdos/students/xcui32/SequentialTraining/datasets/cifar10/train'],
                   'test_annotation_path': None,
                   'test_image_path': ['/u/erdos/students/xcui32/SequentialTraining/datasets/cifar10/test']}

dataset_Imagenet = {'train_annotation_path': ['/u/erdos/cnslab/imagenet-bndbox/bndbox'],
                    'train_image_path': ['/u/erdos/cnslab/imagenet-distinct'],
                    'test_annotation_path': ['/u/erdos/cnslab/imagenet-bndbox/bndbox'],
                    'test_image_path': ['/u/erdos/cnslab/imagenet-distinct']}

if args.dataset_name == 'VOC':
    dataset_paths = dataset_VOC
elif args.dataset_name == 'Imagenet':
    dataset_paths = dataset_Imagenet
elif args.dataset_name == 'CIFAR10':
    dataset_paths = dataset_CIFAR10
elif args.dataset_name == 'COCO':
    raise Exception(f"{args.dataset_name} has not been implemented")
else:
    raise Exception(f"{args.dataset_name} has not been implemented")

all_regimens = ['stb_endsame', 'bts_startsame', 'llo', 'random_oneseq', 'random1', 'single']

optimizer_kwargs = {'lr': args.lr, 'momentum': 0.9}
scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': args.lr_patience, 'min_lr': args.min_lr}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
config = Config(regimens=all_regimens if (args.regimens is None) or (args.regimens == 'all') else [x.strip() for x in
                                                                                                   args.regimens.split(
                                                                                                       ',')],
                dataset_name=args.dataset_name,
                train_image_path=dataset_paths['train_image_path'],
                train_annotation_path=dataset_paths['train_annotation_path'],
                test_annotation_path=dataset_paths['test_annotation_path'],
                test_image_path=dataset_paths['test_image_path'],
                num_samples_to_use=args.num_samples_to_use, test_size=args.test_size,
                num_classes=args.num_classes, cls_to_use=args.cls_to_use,
                sizes=[eval(x.strip()) for x in args.sizes.split(',')],
                input_size=(args.input_size, args.input_size),
                resize_method=args.resize_method,
                model=args.model,
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


def train_regimen(regimen: str, train_indices: Optional[List[int]] = None, test_indices: Optional[List[int]] = None):
    regimen = get_regimen_dataloaders(input_size=config.input_size, sizes=config.sizes, regimen=regimen, num_samples_to_use=config.num_samples_to_use,
                                      dataset_name=config.dataset_name, image_roots=config.train_image_path,
                                      train_indices=train_indices)

    # get all test dataloaders
    test_dataloaders = []
    for size in sorted(config.sizes):
        test_dataset = get_dataset(dataset_name=config.dataset_name, size=config.input_size, p=size,
                                   image_roots=config.test_image_path, num_samples_to_use=config.num_samples_to_use,
                                   indices=test_indices, annotation_roots=config.test_annotation_path,
                                   resize_method=config.resize_method, train=False,
                                   cls_to_use=config.cls_to_use, num_classes=config.num_classes)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True,
                                     num_workers=config.num_workers, pin_memory=True)
        test_dataloaders.append(test_dataloader)

    criterion = criterion_object()
    records = []

    test_result = []

    for fold, sequence in enumerate(regimen):

        model_best_states = {}
        record_dict = {}
        test_df = pd.DataFrame(index=sorted(config.sizes))
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
            model = model.to(device)
            epochs_per_size = int(np.ceil(config.epoch / len(sequence_dataloaders)))
            sequence_list = eval(sequence_name)  # [0.6, 0.8, 1]
            record_list = []
            for seq_idx, (train_dataloader, val_dataloader) in enumerate(sequence_dataloaders):
                print(f'# of trains: {len(train_dataloader) * config.batch_size} # of vals: {len(val_dataloader) * config.batch_size}')
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

                    print(f'==> Current group: {sequence_list[seq_idx]}')
                    trainer = Trainer(criterion=criterion, patience=config.early_stop_patience, device=config.device)
                    for epoch in range(epochs_per_size):

                        train_loss, train_acc, val_loss, val_acc, lr = trainer.train(epoch, model, train_dataloader,
                                                                                     val_dataloader, optimizer)

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
                    model.load_state_dict(model_best_states[str(sequence_list[:seq_idx + 1])])
                    print(
                        f"Group: {sequence_list[seq_idx]} finished training. Best epoch: {trainer.best_epoch + 1} "
                        f"Best val accuracy: {trainer.best_val_acc} Best val loss: {trainer.best_val_loss}")
                    print('\n')
            record_dict[sequence_name] = record_list

            # testing
            test_list = []
            for test_dataloader in test_dataloaders:
                acc_1 = 0
                with torch.no_grad():
                    for images, labels in test_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)

                        acc = metrics.accuracy(outputs, labels, (1,))
                        acc_1 += acc[0]

                    acc_1 = acc_1 / len(test_dataloader)
                    acc_1 = float(acc_1.cpu().numpy()[0])

                test_list.append(acc_1)
            test_df[sequence_name] = test_list

        test_result.append(test_df)
        records.append(record_dict)

    print("Test results:")
    for fold, df in enumerate(test_result):
        print(f"Fold {fold}")
        print(df)
    result = {'config': config,
              'train': records,
              'test': test_result}
    with open(os.path.join(config.result_path, 'result'), 'wb') as handle:
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
    if config.dataset_name == 'Imagenet':

        np.random.seed(config.random_seed)
        indices = list(range(num_samples))
        train_indices = indices[:int((1 - config.test_size) * len(indices))]
        test_indices = indices[int((1 - config.test_size) * len(indices)):]
        for regimen in config.regimens:
            print(f'==>Training {regimen}\n')
            train_regimen(regimen=regimen, train_indices=train_indices, test_indices=test_indices)
            print('\n')

    else:
        for regimen in config.regimens:
            print(f'==>Training {regimen}\n')
            train_regimen(regimen=regimen)
            print('\n')


if __name__ == '__main__':
    main()

