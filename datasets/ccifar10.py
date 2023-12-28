import os
import random
import numpy as np

import torch
from torchvision import datasets, transforms


class CIFAR10(datasets.CIFAR10):
    name = 'CIFAR10'
    num_classes = 10

    def __init__(self, args, train=True):
        self.split_data = []
        self.split_target = []
        self.root = os.path.join(args.root, args.dataset)

        transform = transforms.Compose(
                                       [transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2470, 0.2435, 0.2615))])


        datasets.CIFAR10.__init__(self, root=self.root, train=train, transform=transform, download=True)

        if train:
            save_path = self.root + '/train'
            if not os.path.exists(save_path): os.mkdir(save_path)
        else:
            save_path = self.root + '/test'
            if not os.path.exists(save_path): os.mkdir(save_path)

        for y in range(self.num_classes):
            cls_idx = torch.nonzero(torch.Tensor(self.targets) == y)
            self.split_data = [self.data[loc] for loc in cls_idx]
            self.split_target = [self.targets[loc] for loc in cls_idx]
        
            np.save(os.path.join(save_path, args.dataset + '_Class' + str(y)), np.array(self.split_data))
            np.save(os.path.join(save_path, args.dataset + '_Labels' + str(y)), np.array(self.split_target))

    def __getitem__(self, index):
        x, y = self.split_data[index], self.split_target[index]
        
        return x, y