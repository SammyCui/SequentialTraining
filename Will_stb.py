# Banks 1978 paper:
# 1 month:  2.4 cyc/deg
# 2 month:  2.8 cyc/deg
# 3 month:  4 cyc/deg
# 224 pixels:
# 20 deg -> 11 pix in deg;  4.6 pix blur;  4 pix blur;  2.8 pix blur
# 4 deg -> 56 pix in deg; 23 pix blur (1 mo); 20 pix blur (2 mo); 14 pix blur (3 mo)
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torchvision.datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import scipy
from torch.utils.data.sampler import SubsetRandomSampler
from data_utils import VOCDistancingImageLoader, GenerateBackground
from torch_datasets import VOCImageFolder

if __name__ == '__main__':

    modelType = 'alexnet'
    numEpochs = 300
    image_set = 'VOC' # 'imagewoof', 'imagenette', 'VOC'
    # block_call = args[4] # int {0:4}

    # Example call:
    # python3 alexnet 100 imagenette 1

    background = GenerateBackground(bg_type='fft')
    size = (150, 150)
    training_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/train"
    train_annotation_root_path = os.path.join(training_root_path, 'annotations')
    cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor']
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor']
    data_dir = os.path.join(training_root_path, 'root')
    num_workers = 2
    def get_train_valid_loader(block, distance, random_seed=69420, valid_size=0.2, shuffle=False,
                               show_sample=False, num_workers=num_workers, pin_memory=False, batch_size=128):
        # valid_size gotta be in [0,1]
        # block must be an int between 0:(1/valid_size) (0:4 for valid_size==0.2)
        transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.Resize((150,150)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        train_loader = VOCDistancingImageLoader(size, p=distance,
                                                background_generator=background,
                                                annotation_root_path=train_annotation_root_path)

        train_dataset = VOCImageFolder(root=data_dir, cls_to_use=cats, loader=train_loader, transform=transform)
        valid_dataset = VOCImageFolder(root=data_dir, cls_to_use=cats, loader=train_loader, transform=transform)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        split1 = int(np.floor(block * split))
        split2 = int(np.floor((block + 1) * split))
        # if shuffle:
        np.random.seed(100)
        np.random.shuffle(indices)
        valid_idx = indices[split1:split2]
        train_idx = np.append(indices[:split1], indices[split2:])
        train_idx = train_idx.astype('int32')

        if block != 0:
            for b in range(block):
                indices = [indices[(i + split) % len(indices)] for i, x in enumerate(indices)]
        # train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        # train_sampler = torch.utils.data.Subset(dataset, indices)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, sampler=valid_sampler, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        return (train_loader, valid_loader)


    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # blurTypes = ['average', 'gaussian', 'pixelate']
    blurTypes = ['gaussian']
    # data_dir = "C:/Users/roaga/OneDrive/Documents/Summer_Internship_2019/imagewoof-320_blur/"
    # data_dir = "/home/cnslab/imagenet_blur/"

    # for directory, subdirectories, files in os.walk(data_dir):
    #        for file in files:
    #              if directory.split("\\")[-1] not in classes:
    #                        classes.append(directory.split("\\")[-1])

    ######
    # hardcoding the classes for sizeNet because I am too lazy to properly debug right now: (5/24)
    # classes = ['frog', 'doberman', 'kit fox', 'crane', 'helmet', 'limo', 'tennis racket', 'flower']


    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device = ' + str(device))


    def train(blrLvl):
        # for epoch in range(int(num_epochs_sub)):  # loop over the dataset multiple times
        adjustedEpochs = int(int(numEpochs) / int(len(blrLvl)))
        for epoch in range(int(adjustedEpochs)):
            prev_loss = 100000.0
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                net.train()
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if epoch % 10 == 9:
                net.eval()
                total = 0
                correct = 0
                val_loss = 0
                with torch.no_grad():
                    for data in validloader:
                        images, labels = data
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = net(images)
                        val_loss += criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        acc = 100 * correct / total


                print('[%d,  validation acc: %5d, validation loss: %.5f training loss: %.5f' %
                      (epoch + 1, acc, val_loss, running_loss / 100))
            # if epoch == (int(adjustedEpochs) - 1):
            #   torch.save(net.state_dict(),'/u/erdos/cnslab/will_witek/VOC_Models/' + modelType + '_' + image_set + '8cats_reverse_epoch' + str(epoch) + '_blur' + str(blurLevels))


    allAccs = []
    for blurType in blurTypes:  # multiple types of blur
        print(blurType)
        print('-' * 10)
        for block in range(5):
            # block = int(block_call)
            print("\nFOLD " + str(block + 1) + ":")
            for i in range(5):
                # for i in (0,4):
                if i == 0:
                    blurLevels = [0.2, 0.25, 0.333, 0.5, 1]
                elif i == 1:
                    blurLevels = [0.25, 0.333, 0.5, 1]
                elif i == 2:
                    blurLevels = [0.333, 0.5, 1]
                elif i == 3:
                    blurLevels = [0.5, 1]
                elif i == 4:
                    blurLevels = [1]

                if modelType == 'vgg16':
                    net = torchvision.models.vgg16(pretrained=False)
                    num_ftrs = net.classifier[6].in_features
                    net.classifier[6] = nn.Linear(num_ftrs, len(classes))
                elif modelType == 'alexnet':
                    net = torchvision.models.alexnet(pretrained=False)
                    num_ftrs = net.classifier[6].in_features
                    net.classifier[6] = nn.Linear(num_ftrs, len(classes))
                else:
                    net = torchvision.models.squeezenet1_1(pretrained=False)
                    net.classifier[1] = nn.Conv2d(512, len(classes), kernel_size=(1, 1), stride=(1, 1))
                    net.num_classes = len(classes)
                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                net = net.to(device)
                for i in range(len(blurLevels)):  # 5 levels of blur: 1, 3, 5, 11, 23
                    mult = blurLevels[i]
                    trainloader, validloader = get_train_valid_loader(block=block, distance=mult, shuffle=False,
                                                                      num_workers=num_workers, batch_size=128)

                    print('Start training on blur window of ' + str(mult))
                    train(blurLevels)
                    print('Finished Training on ' + blurType + ' with blur window of ' + str(mult))

                accs = []
                permBlurLevels = [0.2, 0.25, 0.333, 0.5, 1]
                for j in range(len(permBlurLevels)):
                    tempMult = permBlurLevels[j]
                    correct = 0
                    total = 0
                    t2, validloader2 = get_train_valid_loader(block=block, distance=tempMult, shuffle=False, num_workers=num_workers,
                                                              batch_size=128)

                    with torch.no_grad():
                        for data in validloader2:
                            images, labels = data
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            acc = 100 * correct / total
                        print('Accuracy: %f %%' % (acc))
                        accs.append(acc)

                    allAccs.append(accs)
