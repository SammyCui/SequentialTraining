import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from utils.data_utils import GenerateBackground
from loader import CIFARLoader
from datasets_legacy import CIFAR10Dataset
from utils import metrics

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=80, type=int, help='num of epochs')
parser.add_argument('--model', default='resnet50', type=str, help='model name from pytorch')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers for dataloader')
parser.add_argument('--lr_patience', default=3, type=int, help='number of epochs to wait before decreasing lr')
parser.add_argument('--min_lr', default=0.00001, type=float, help='minimum learning rate')
parser.add_argument('--n_folds', default=5, type=int, help='number of folds')
parser.add_argument('--n_folds_to_use', default=1, type=int, help='number of folds to use')

# data path
parser.add_argument('--path', default='./datasets/cifar10', help='path to cifar dataset')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device: ', device)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_root, test_root = os.path.join(args.path, 'train'), os.path.join(args.path, '')

train_loader = CIFARLoader(size=(32, 32), p=1,
                           background_generator=GenerateBackground(bg_type='color', bg_color=(0, 0, 0)),
                           resize_method='long')
train_dataset = CIFAR10Dataset(root=train_root,
                               transform=transform_train,
                               loader=None,  #train_loader,
                               download=True,
                               train=True)

test_loader = CIFARLoader(size=(32, 32), p=1,
                          background_generator=GenerateBackground(bg_type='color', bg_color=(0, 0, 0)),
                          resize_method='long')
test_dataset = CIFAR10Dataset(root=test_root,
                              transform=transform_train,
                              loader=None,  #test_loader,
                              download=True,
                              train=False)
test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)

training_len = len(train_dataset)
num_train = training_len
val_size = 1 / args.n_folds

cv_indices = list(range(num_train))
np.random.seed(40)
np.random.shuffle(cv_indices)

split = int(np.floor(val_size * num_train))


# Training
def train(model, dataloader, optimizer):
    model.train()
    training_loss = 0
    acc = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        batch_acc = metrics.accuracy(outputs, targets, (1, 5))
        acc += batch_acc[0]
    acc /= len(dataloader)
    acc = float(acc.cpu().numpy()[0])
    return training_loss, acc


def val(model, dataloader):
    model.eval()
    val_loss = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_acc = metrics.accuracy(outputs, targets, (1, 5))
            acc += batch_acc[0]
        acc /= len(dataloader)
        acc = float(acc.cpu().numpy()[0])

    return val_loss, acc


fold_model_list = []

for fold in range(args.n_folds):

    if args.n_folds_to_use:
        if fold >= args.n_folds_to_use:
            break

    print('fold ', fold)
    split1 = int(np.floor(fold * split))
    split2 = int(np.floor((fold + 1) * split))
    val_idx = cv_indices[split1:split2]
    train_idx = np.append(cv_indices[:split1], cv_indices[split2:])
    train_idx = train_idx.astype('int32')
    train_size = 5000  # num_train
    train_idx = train_idx[:train_size]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    print('Training length ', len(train_idx))
    print('Validating length ', len(val_idx))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=100, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    net = eval('torchvision.models.' + args.model + f'(num_classes=10)')
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    sgd = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, min_lr=args.min_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sgd, T_max=args.epoch)

    best_val_loss = np.inf

    for epoch in range(args.epoch):
        train_loss, train_acc = train(net, train_dataloader, sgd)
        validation_loss, validation_acc = val(net, val_dataloader)

        if validation_loss < best_val_loss:
            best_state = net.state_dict()
            best_epoch = epoch + 1
            best_val_acc = validation_acc
            best_val_loss = validation_loss

        print("Epoch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f} Val Acc@1: {:.3f} lr: {:.5f} best_epoch: {}, best_val_loss: {:.4f}"
              .format(epoch + 1, args.epoch, train_loss, validation_loss, validation_acc,
                      sgd.state_dict()['param_groups'][0]['lr'], best_epoch, best_val_loss))
        scheduler.step()

    net = eval('torchvision.models.' + args.model + f'(num_classes=10)')
    net.load_state_dict(best_state)
    net.to(device)
    test_loss, test_acc = val(net, test_dataloader)
    print(f'Best epoch: {best_epoch}')
    print(f'Best validation acc: {best_val_acc}')
    print('test acc: ', test_acc)

    print('-' * 50)




