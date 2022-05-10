import torchvision
import torch

"""
Modify output num_classes and input num_channels of the torch models. Full doc:
    https://github.com/pytorch/vision/blob/1db8795733b91cd6dd62a0baa7ecbae6790542bc/torchvision/models/resnet.py#L286

:param pretrained:
:param num_classes: number of output classes 

"""


def resnet18(num_classes: int, pretrained: bool = False):
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.num_classes = num_classes
    model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model


def alexnet(num_classes: int, pretrained: bool = False):
    model = torchvision.models.alexnet(pretrained=pretrained)
    model.num_classes = num_classes
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    return model


def squeezenet1_1(num_classes: int, pretrained: bool = False):
    model = torchvision.models.squeezenet1_1(pretrained=pretrained)
    model.num_classes = num_classes
    model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


def mobilenet_v3_small(num_classes: int, pretrained: bool = False):
    model = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
    model.num_classes = num_classes
    model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    return model
