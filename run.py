from model_training.pipeline import RunModel
from data_utils import GenerateBackground, VOCDistancingImageLoader, VOCImageFolder
import os
from pathlib import Path
import torch, torchvision


def run(mode: str, bg: str):
    cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor']
    train_root_path = "/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/train"
    val_root_path = "/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/val"
    test_root_path = "/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/test"
    result_dirpath = "/u/erdos/students/xcui32/cnslab/results"

    pipeline_params = {'train_root_path': train_root_path,
                       'val_root_path': val_root_path,
                       'test_root_path': test_root_path,
                       'dataset_name': 'VOC',
                       'target_distances': [0.2,0.4,0.6,0.8, 1],
                       'training_mode': mode,
                       'n_distances': None,
                       'background': GenerateBackground(bg_type=bg),
                       'size': (150, 150),
                       'cls_to_use': None,
                       'epochs': 100,
                       'val_size': 0.8,
                       'batch_size': 64,
                       'num_workers': 8,
                       'model_name': 'alexnet',
                       'result_dirpath': result_dirpath,
                       'random_seed': 40,
                       'save_checkpoints': False}

    criterion_object = torch.nn.CrossEntropyLoss
    optimizer_object = torch.optim.Adam

    optimizer_kwargs = {'lr': 0.001, 'weight_decay': 1e-4}
    VOCpipeline = RunModel(**pipeline_params)
    print(' ---  Loading datasets ---')
    VOCpipeline.load_datasets()
    print(' ---  Running  ---')
    VOCpipeline.run(criterion_object=criterion_object, optimizer_object=optimizer_object, patience=10, 
                    **optimizer_kwargs)
    print(' --- Evaluating ---')
    VOCpipeline.evaluate()


if __name__ == '__main__':
    modes = ['stb','bts','random','llo','single']
    bgs = ['color']
    for mode in modes:
        for bg in bgs:
            print(f' # ------------------ Running pipeline on {mode} {bg} -------------------- #')
            run(mode,bg)
            print('-' * 30 + ' end ' + '-' * 30)
            print('\n')
