import os
import sys

sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages')
sys.path.append('/u/erdos/students/xcui32/.local/lib/python3.6/site-packages')
print(sys.path)

import torch, torchvision
from pipeline_cv import RunModel
from data_utils import GenerateBackground
from pathlib import Path


def main(mode, bg, suffix):
    cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor']
    train_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/train"
    val_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/val"
    test_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/test"
    result_dirpath = Path(__file__).parent / "results/VOC8AlexnetCVavg"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    pipeline_params = {'train_root_path': train_root_path,
                       'val_root_path': val_root_path,
                       'test_root_path': test_root_path,
                       'dataset_name': 'VOC',
                       'target_distances': [0.2,0.4,0.6, 0.8, 1],
                       'training_mode': mode,
                       'n_distances': None,
                       'background': GenerateBackground(bg_type=bg, bg_color=(0,0,0)),
                       'size': (150, 150),
                       'device': device,
                       'cls_to_use': cats,
                       'verbose': 0,
                       'resize_method': 'long',
                       'epochs': 400,
                       'n_folds': 5,
                       'batch_size': 128,
                       'num_workers': 16,
                       'model_name': 'alexnet',
                       'result_dirpath': result_dirpath,
                       'random_seed': 40,
                       'save_checkpoints': False}

    criterion_object = torch.nn.CrossEntropyLoss
    #optimizer_object = torch.optim.Adam
    optimizer_object = torch.optim.SGD

    #optimizer_kwargs = {'lr': 0.001, 'weight_decay': 1e-4}
    optimizer_kwargs = {'lr': 0.001, 'momentum': 0.9}
    VOCpipeline = RunModel(**pipeline_params)
    print(' ---  Loading datasets ---')
    VOCpipeline.load_datasets()
    print(' ---  Running  ---')
    VOCpipeline.run(criterion_object=criterion_object, optimizer_object=optimizer_object, patience=20, val_target='avg',
                    optim_kwargs=optimizer_kwargs)
    print('\n')
    print(' --- Evaluating ---')
    VOCpipeline.evaluate(suffix=suffix)


if __name__ == '__main__':
    modes = ['bts_startsame', 'stb_endsame', 'single', 'random', 'llo']
    modes2 = ['bts_startsame', 'stb_endsame','random','llo']
    bgs = ['fft']
    num_of_runs = 1
    for run in range(num_of_runs):
        print('Run: ', run)
        for mode in modes:
            for bg in bgs:
                print(f' # ------------------ Running pipeline on {mode} {bg} run_{run} -------------------- #')
                main(mode, bg, suffix=run)
                print('-' * 30 + ' End ' + '-' * 30)
                print('\n')
        print('\n')
        print('\n')
        print('\n')