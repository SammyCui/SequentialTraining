from typing import Callable, Tuple, List, Dict

import torchvision

from .utils import metrics

import torch

from torch.utils.data import DataLoader


def evaluate(model, test_dataloaders: List[(float, DataLoader)], device: str = 'cpu'):
    result_dict = {}
    model.eval()

    for size, test_dataloader in test_dataloaders:

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

        result_dict[str(size)] = acc_1

    return result_dict



