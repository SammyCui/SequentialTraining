from typing import List, Optional, Tuple, Iterable, Union
import torch
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler, ConcatDataset, DataLoader, Subset
from .helpers import get_dataset, get_cv_indices


def get_regimen_dataloaders(input_size: Tuple[int, int], sizes: List[float], regimen: str,
                            dataset_name: str,
                            image_roots: List[str],
                            train_indices: List[int] = None,
                            annotation_roots: List[str] = None,
                            resize_method: str = 'long',
                            n_folds: int = 5,
                            n_folds_to_use: int = 5,
                            num_samples_to_use: int = None,
                            random_seed: int = 40,
                            batch_size: int = 128,
                            num_workers: int = 16,
                            **kwargs):
    """

    :param n_folds_to_use:
    :param resize_method:
    :param train_indices:
    :param num_samples_to_use:
    :param input_size:
    :param sizes:
    :param regimen:
    :param dataset_name:
    :param image_roots:
    :param annotation_roots:
    :param n_folds:
    :param random_seed:
    :param batch_size:
    :param num_workers:
    :param kwargs:
    :return: in the format of: [
                                [("[0.8, 1]", [(train_dataloader_0.8, val_dataloader_0.8), (train_dataloader_1, val_dataloader_1)]),
                                 ("[0.6, 0.8, 1]", [(), (), ()]),
                                 ("[0.4, 0.6, 0.8, 1]", [(), (), (), ()]),
                                 ("[0.2, 0.4, 0.6, 0.8, 1]", [(), (), (), (), ()])],            # fold 1
                                [(sequence_name, [(), (), ()]), (sequence_name, [(), (), ()])], # fold2 ...
                               ]
    """

    fold_list = []

    if regimen == 'random_1group':
        dataset_list = []
        for size in sizes:
            d = get_dataset(dataset_name=dataset_name, size=input_size, p=size, image_roots=image_roots,
                            resize_method=resize_method, annotation_roots=annotation_roots, indices=train_indices,
                            **kwargs)
            dataset_list.append(d)
        dataset_all_size = torch.utils.data.ConcatDataset(dataset_list)

        # Increase number of samples to use since we only have one training group with all sizes mixed.
        if num_samples_to_use: num_samples_to_use = num_samples_to_use * len(sizes)

        for train_idx, val_idx in get_cv_indices(num_samples=len(dataset_all_size),
                                                 num_samples_to_use=num_samples_to_use, n_folds=n_folds,
                                                 n_folds_to_use=n_folds_to_use, random_seed=random_seed):
            train_dataloader = DataLoader(dataset_all_size, batch_size=batch_size,
                                          sampler=SubsetRandomSampler(train_idx),
                                          num_workers=num_workers, pin_memory=True)
            val_dataloader = DataLoader(dataset_all_size, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx),
                                        num_workers=num_workers, pin_memory=True)
            fold_list.append([(str(['random']), [(train_dataloader, val_dataloader)])])

    elif regimen == 'random':
        dataset_list = []
        for size in sizes:
            d = get_dataset(dataset_name=dataset_name, size=input_size, p=size, image_roots=image_roots,
                            resize_method=resize_method, annotation_roots=annotation_roots, indices=train_indices,
                            **kwargs)
            dataset_list.append(d)
        dataset_all_size = torch.utils.data.ConcatDataset(dataset_list)
        indices = np.arange(len(dataset_all_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        indices_sequence = np.array_split(indices, len(sizes))
        for train_idx, val_idx in get_cv_indices(num_samples=len(indices_sequence[0]),
                                                 num_samples_to_use=num_samples_to_use, n_folds=n_folds,
                                                 n_folds_to_use=n_folds_to_use, random_seed=random_seed):
            sequence = []
            for indices in indices_sequence:
                dataset = Subset(dataset_all_size, indices)
                train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx),
                                              num_workers=num_workers, pin_memory=True)
                val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx),
                                            num_workers=num_workers, pin_memory=True)
                sequence.append((train_dataloader, val_dataloader))
            fold_list.append([(str([f'random_{i}' for i in range(len(sizes))]), sequence)])

    elif regimen == 'random-single':
        dataset_sequence_list = []
        for i in range(len(sizes)):

            # mix all sizes except the single size at the end
            mixed_sizes = sizes[:i] + sizes[i + 1:]
            dataset_list = []
            for size in mixed_sizes:
                d = get_dataset(dataset_name=dataset_name, size=input_size, p=size, image_roots=image_roots,
                                resize_method=resize_method, annotation_roots=annotation_roots, indices=train_indices,
                                **kwargs)
                dataset_list.append(d)
            dataset_all_size = torch.utils.data.ConcatDataset(dataset_list)
            indices = np.arange(len(dataset_all_size))
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            indices_sequence = np.array_split(indices, len(sizes) - 1)
            single_size_dataset = get_dataset(dataset_name=dataset_name, size=input_size, p=sizes[i],
                                              image_roots=image_roots,
                                              resize_method=resize_method, annotation_roots=annotation_roots,
                                              indices=train_indices, **kwargs)
            dataset_sequence_list.append(
                [Subset(dataset_all_size, indices) for indices in indices_sequence] + [single_size_dataset])

        for train_idx, val_idx in get_cv_indices(num_samples=len(dataset_sequence_list[0][0]),
                                                 num_samples_to_use=num_samples_to_use, n_folds=n_folds,
                                                 n_folds_to_use=n_folds_to_use, random_seed=random_seed):
            fold_sequence = []
            for size_idx, dataset_sequence in enumerate(dataset_sequence_list):
                dataloader_sequence = []
                for dataset in dataset_sequence:
                    train_dataloader = DataLoader(dataset, batch_size=batch_size,
                                                  sampler=SubsetRandomSampler(train_idx),
                                                  num_workers=num_workers, pin_memory=True)
                    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx),
                                                num_workers=num_workers, pin_memory=True)
                    dataloader_sequence.append((train_dataloader, val_dataloader))
                sequence_str = str([f'llo_random_{i}_{sizes[size_idx]}' for i in range(len(sizes) - 1)] + [
                    str(sizes[size_idx])])  # e.g. '["llo_random0_0.2", "llo_random1_0.2"..., '0.2']'
                fold_sequence.append((sequence_str, dataloader_sequence))
            fold_list.append(fold_sequence)

    else:

        if regimen == 'stb_endsame':
            sizes = sorted(sizes)
            sequences = [sizes[i:] for i in range(len(sizes) - 1)]

        elif regimen == 'stb_startsame':
            sizes = sorted(sizes)
            sequences = [sizes[:i + 1] for i in range(1, len(sizes))]

        elif regimen == 'bts_startsame':
            sizes = sorted(sizes, reverse=True)
            sequences = [sizes[:i + 1] for i in range(1, len(sizes))]

        elif regimen == 'bts_endsame':
            sizes = sorted(sizes, reverse=True)
            sequences = [sizes[i:] for i in range(len(sizes) - 1)]

        elif regimen == 'single':
            sequences = [[i] for i in sorted(sizes)]

        elif regimen == 'as_is':
            sequences = [sizes]

        else:
            raise Exception(f'Unknown regimen: {regimen}')

        if train_indices:
            num_samples = len(train_indices)
        else:
            num_samples = len(get_dataset(dataset_name=dataset_name, size=input_size, p=1, image_roots=image_roots,
                                          resize_method=resize_method, annotation_roots=annotation_roots,
                                          indices=train_indices, **kwargs))
        for train_idx, val_idx in get_cv_indices(num_samples=num_samples,
                                                 num_samples_to_use=num_samples_to_use, n_folds=n_folds,
                                                 n_folds_to_use=n_folds_to_use, random_seed=random_seed):
            fold_sequence = []
            for sequence in sequences:
                dataloader_sequence = []
                for size_group in sequence:
                    dataset = get_dataset(dataset_name=dataset_name, size=input_size, p=size_group,
                                          image_roots=image_roots, annotation_roots=annotation_roots,
                                          indices=train_indices, **kwargs)
                    train_dataloader = DataLoader(dataset, batch_size=batch_size,
                                                  sampler=SubsetRandomSampler(train_idx),
                                                  num_workers=num_workers, pin_memory=True)
                    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx),
                                                num_workers=num_workers, pin_memory=True)
                    dataloader_sequence.append((train_dataloader, val_dataloader))

                fold_sequence.append((str(sequence), dataloader_sequence))

            fold_list.append(fold_sequence)

    return fold_list


if __name__ == '__main__':
    train_val_dataloaders = get_regimen_dataloaders(input_size=(150, 150), sizes=[0.2, 0.4, 1], regimen='stb_startsame',
                                                    dataset_name='VOC',
                                                    image_roots=[
                                                        './datasets/VOC2012_filtered/train/root',
                                                        './datasets/VOC2012_filtered/val/root'],
                                                    train_indices=None,
                                                    annotation_roots=[
                                                        './datasets/VOC2012_filtered/train/annotations',
                                                        './datasets/VOC2012_filtered/val/annotations'],
                                                    n_folds=5,
                                                    num_samples_to_use=5000,
                                                    random_seed=40,
                                                    batch_size=128,
                                                    num_workers=0,
                                                    )

    print()
