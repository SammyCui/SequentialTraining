import torch, torchvision
from data_utils import GenerateBackground, VOCImageFolder, VOCDistancingImageLoader
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

def test():
    num_workers = 6
    transform = torchvision.transforms.Compose([

        torchvision.transforms.ToTensor()
        ,

        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    distance = 0.1


    train_loader = VOCDistancingImageLoader((150, 150), distance,
                                            background_generator=GenerateBackground(bg_type='fft'),
                                            annotation_root_path='/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/train/annotations')

    val_loader = VOCDistancingImageLoader((150, 150), distance,
                                          background_generator=GenerateBackground(bg_type='fft'),
                                          annotation_root_path='/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/val/annotations')

    test_loader = VOCDistancingImageLoader((150, 150), distance,
                                           background_generator=GenerateBackground(bg_type='fft'),
                                           annotation_root_path='/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/test/annotations')

    train_dataset = VOCImageFolder(
        cls_to_use=['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor'],
        root='/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/train/root',
        loader=train_loader,
        transform=transform)
    count = torch.tensor(list(Counter(train_dataset.targets).values()))
    weights = 1 / count
    samples_weights = weights[train_dataset.targets]
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights),
                                                     replacement=True)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=64
                                                   , shuffle=True
                                                   # , sampler=sampler
                                                   , num_workers=num_workers,
                                                   sampler=torch.utils.data.DistributedSampler(train_dataset)
                                                   )

    val_dataset = VOCImageFolder(
        cls_to_use=['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor'],
        root='/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/val/root',
        loader=val_loader,
        transform=transform)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=64,
                                                 # sampler=torch.utils.data.RandomSampler(data_source=val_dataset,
                                                 #                                        num_samples=200,
                                                 #                                        replacement=True)
                                                 num_workers=num_workers,
                                                 sampler=torch.utils.data.DistributedSampler(val_dataset))

    test_dataset = VOCImageFolder(
        cls_to_use=['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor'],
        root='/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/test/root',
        loader=test_loader,
        transform=transform)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=64,
                                                  shuffle=True
                                                  , num_workers=num_workers,
                                                  sampler=torch.utils.data.DistributedSampler(test_dataset))

    patience = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_dataset.classes)
    model = torchvision.models.alexnet(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    best_sub_model, best_sub_optimizer = None, None
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    all_training_loss = []
    patience_count = 0
    best_epoch = 0
    best_val_acc = 0
    best_val_loss = np.inf
    sub_training_loss, sub_val_loss = [], []
    val_top1_acc = []
    epochs = 100
    # TODO: validation set for separate distances
    best_loss = np.inf
    for epoch in range(epochs):

        # training
        model.train()
        training_loss_per_pass = 0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss_per_pass += loss.item()

        sub_training_loss.append(training_loss_per_pass)

        # validation
        correct = 0
        total = 0
        val_loss_per_pass = 0
        acc = -1
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss = criterion(outputs, labels)
                val_loss_per_pass += val_loss.item()
                acc = correct / total
                print('-' * 40)
                print('predicted: ', predicted)
                print('\n')
                print('labels: ', labels)
                print('-' * 40)
                print('\n')

        val_top1_acc.append(acc)
        sub_val_loss.append(val_loss_per_pass)
        print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Top1 '
              'Accuracy: {:.4f}'
              .format(epoch + 1, epochs, training_loss_per_pass, val_loss_per_pass, acc))
        if val_loss_per_pass <= best_val_loss:
            print('update best val ', val_loss_per_pass)
            best_val_loss = val_loss_per_pass
            best_sub_model = model.state_dict()
            best_sub_optimizer = optimizer.state_dict()
            patience_count = 0
            best_epoch = epoch
            best_val_acc = acc

if __name__ =='__main__':
    test()