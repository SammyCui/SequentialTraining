import ctypes
# libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import os
import sys

sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages')
sys.path.append('/u/erdos/students/xcui32/.local/lib/python3.6/site-packages')

import torch, torchvision
from pipelineCV2 import RunModel
from data_utils import GenerateBackground
from pathlib import Path


def main(mode, bg, suffix):
    cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor']
    train_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/train"
    val_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/val"
    test_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/test"

    train_root_path_imgnt = Path(__file__).parent / "datasets/imagenette-160/imagenette2-160/train"
    test_root_path_imgnt = Path(__file__).parent / "datasets/imagenette-160/imagenette2-160/val"

    result_dirpath = Path(__file__).parent / "results/VOC20-5000R18stb1" 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    testing_distances = [0.2,0.4,0.6,0.8,1]
    print(device)
    pipeline_params = {'train_root_path': train_root_path,
                       'val_root_path': val_root_path,
                       'test_root_path': test_root_path,
                       'dataset_name': 'VOC',                      
                       'target_distances': [1,0.2, 0.4, 0.6, 0.8, 1],
                       'testing_distances': testing_distances,
                       'training_mode': mode,
                       'background': GenerateBackground(bg_type=bg, bg_color=(0, 0, 0)),
                       'size': (150, 150),
                       'training_size': 5000,
                       'n_folds_to_use':2,
                       'device': device,
                       'cls_to_use': None,#cats,
                       'verbose': 0,
                       'resize_method': 'long',
                       'epochs': 200,
                       'n_folds': 5,
                       'batch_size': 128,
                       'num_workers': 16,
                       'model_name': 'resnet18',
                       'result_dirpath': result_dirpath,
                       'random_seed': 40,
                       'save_checkpoints': False,
                       'save_progress_checkpoints':False}

    optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9}

    scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5}
    fit_params = {'criterion_object': torch.nn.CrossEntropyLoss,
                  'optimizer_object': torch.optim.SGD,
                  'scheduler_object': torch.optim.lr_scheduler.ReduceLROnPlateau,
                  'patience': 20,
                  'reset_lr':None,
                  'max_norm': None,#2,
                  'val_target': 'current',
                  'optim_kwargs': optimizer_kwargs,
                  'scheduler_kwargs': scheduler_kwargs}

    VOCpipeline = RunModel(**pipeline_params)
    print(' ---  Loading datasets ---')
    VOCpipeline.load_datasets()
    print(' ---  Running  ---')
    VOCpipeline.run(**fit_params)
    print('\n')
    print(' --- Evaluating ---')
    VOCpipeline.evaluate()


if __name__ == '__main__':
    modes = ['bts_startsame','stb_endsame', 'llo', 'random_oneseq', 'single']
    modes2 = ['as_is']
    modes3 = ['random_oneseq']
    bgs = ['color']
    num_of_runs = 1
    for run in range(num_of_runs):
        print('Run: ', run)
        for mode in modes2:
            for bg in bgs:
                print(f' # ------------------ Running pipeline on {mode} {bg} run_{run} -------------------- #')
                main(mode, bg, suffix=run)
                print('-' * 30 + ' End ' + '-' * 30)
                print('\n')
        print('\n')
        print('\n')
        print('\n')
