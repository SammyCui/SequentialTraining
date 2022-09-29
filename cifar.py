# libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import sys
from typing import List

sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages')
sys.path.append('/u/erdos/students/xcui32/.local/lib/python3.6/site-packages')

import torch
from pipeline_cifar10 import RunModel
from utils.data_utils import GenerateBackground
from pathlib import Path


def main(modes: List[str]):
    train_root_path = "./datasets/cifar10/train"
    test_root_path = "./datasets/cifar10/test"
    result_dirpath = Path(__file__).parent / "results/Cifar10R18Black"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    testing_distances_all = [0.2, 0.4, 0.6, 0.8, 1]
    print(device)
    pipeline_params = {'train_root_path': train_root_path,
                       'test_root_path': test_root_path,
                       'dataset_name': 'cifar10',
                       'target_distances': [1],
                       'testing_distances': [1],
                       'background': GenerateBackground(bg_type='color', bg_color=(0, 0, 0)),
                       'size': (32, 32),
                       'device': device,
                       'llo_targets':None,
                       'n_folds_to_use':3,
                       'verbose': 0,
                       'resize_method': 'long',
                       'epochs': 1,
                       'n_folds': 5,
                       'batch_size': 128,
                       'num_workers': 2,
                       'model_name': 'resnet18',
                       'result_dirpath': result_dirpath,
                       'random_seed': 40,
                       'save_checkpoints': False,
                       'save_progress_checkpoints': False}

    optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9}

    scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5}
    fit_params = {'criterion_object': torch.nn.CrossEntropyLoss,
                  'optimizer_object': torch.optim.SGD,
                  'scheduler_object': torch.optim.lr_scheduler.ReduceLROnPlateau,
                  'patience': 10,
                  'reset_lr': None,
                  'max_norm': 2,
                  'val_target': 'current',
                  'optim_kwargs': optimizer_kwargs,
                  'scheduler_kwargs': scheduler_kwargs}

    for mode in modes:
        pipeline_params['training_mode'] = mode
        print('-' * 30, f'Running on {mode}', '-' * 30)

        pipeline = RunModel(**pipeline_params)
        print('-' * 30, f'Loading Dataset', '-' * 30)
        pipeline.load_datasets()
        print('-' * 30, f'Running', '-' * 30)
        pipeline.run(**fit_params)
        print('\n')
        print('-' * 30, f'Testing', '-' * 30)
        pipeline.evaluate()
        print('-' * 30, f'Done', '-' * 30)


if __name__ == '__main__':
    modes = ['bts_startsame', 'stb_endsame', 'llo', 'single', 'random_oneseq']
    modes2 = ['as_is']
    modes3 = ['random_oneseq', 'llo']

    main(modes2)