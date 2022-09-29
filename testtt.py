import torch, torchvision
from pipeline_imagenet import RunModel
from data_utils import GenerateBackground
from pathlib import Path


def main(mode, bg, suffix):
    train_root_path = "/u/erdos/cnslab/imagenet"
    train_anno_root_path = "/u/erdos/cnslab/imagenet-bndbox/bndbox"


    result_dirpath = Path(__file__).parent / "results/Imagenet40BlackR18long"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    testing_distances_all = [0.2,0.4,0.6,0.8,1]
    print(device)
    pipeline_params = {'train_root_path': train_root_path,
                       'train_anno_path':train_anno_root_path,
                       'dataset_name': 'imagenet',
                       'num_classes': 1,
                       'target_distances': [0.2, 0.4, 0.6, 0.8, 1],
                       'testing_distances': testing_distances_all,
                       'test_size': 0.3,
                       'training_mode': mode,
                       'background': GenerateBackground(bg_type=bg, bg_color=(0, 0, 0)),
                       'size': (150, 150),
                       'training_size': 10000, #None,
                       'device': device,
                       'cls_to_use': None,
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
                  'max_norm': 2,
                  'val_target': 'current',
                  'optim_kwargs': optimizer_kwargs,
                  'scheduler_kwargs': scheduler_kwargs}
    print(' --- Initializing ... ---')
    pipeline = RunModel(**pipeline_params)

if __name__ == '__main__':
    modes = ['bts_startsame','stb_endsame','single', 'random', 'llo']
    modes2 = ['stb_endsame']
    modes3 = ['single','random','llo']
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

