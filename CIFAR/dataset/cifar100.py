from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import sys

from torchvision import datasets, transforms

"""
100 classes
Training data: 50000, 500 images per class
Testing data: 10000,  100 images per class

mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""

def get_augment_transform(jitter_strength=0.5, gaussian_blur=False):
    color_jitter = transforms.ColorJitter(0.8 * jitter_strength, 0.8 * jitter_strength, 
                                    0.8 * jitter_strength, 0.2 * jitter_strength)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    tfs = [
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        color_distort,
    ]
    if gaussian_blur:
        blur = transforms.GaussianBlur(3) # sigma default is correct
        tfs.append(transforms.RandomApply([blur], p=0.5))
    tfs = tfs + [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    return transforms.Compose(tfs)

def get_heavy_augment_transform():
    auto = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        auto,
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
# augment_transform = get_augment_transform(gaussian_blur=True)
augment_transform = get_heavy_augment_transform()
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])



class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    100 classes
    Training data: 50000, 500 images per class
    Testing data: 10000,  100 images per class
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0, no_labels=False): 
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
                
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if self.train:
            num_samples = len(self.data)
            label = self.targets
        else:
            num_samples = len(self.test_data)
            label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes + 1)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])
        self.cls_negative[-1] = np.arange(num_samples)
    

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive) # shape: (100, 500)
        self.cls_negative = np.asarray(self.cls_negative) # shape: (100, 49500)
        self.no_labels = no_labels
     

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            if self.no_labels:
                neg_idx = np.random.choice(self.cls_negative[-1], self.k, replace=replace) 
            else:
                neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace) 
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx)) 
            return img, target, index, sample_idx



class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
class CIFAR100Augment(datasets.CIFAR100):
    """CIFAR100DataAugmentation Dataset.
    """
    def __init__(self, num_transforms=3, **kwargs):
        super().__init__(**kwargs)
        self.num_transform = num_transforms

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        base_img = Image.fromarray(img)
        imgs = []
        if self.transform is not None:
            for _ in range(self.num_transform):
                img = self.transform(base_img)
                imgs.append(img)
        else:
            return NotImplementedError

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs, target


def get_cifar100_dataloaders(data_folder, is_instance=False, self_supervised=False, is_augment=False, num_transforms=3, agg_trans=False):
    """
    cifar 100
    """
    if self_supervised:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=transforms.ToTensor())
        test_set = datasets.CIFAR100(root=data_folder,
                        download=True,
                        train=False,
                        transform=transforms.ToTensor())
        return train_set, test_set
    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
    elif is_augment:
        train_set = CIFAR100Augment(num_transforms=num_transforms, 
                                    root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=get_heavy_augment_transform())
    else:
        trans = train_transform
        if agg_trans:
            trans = get_heavy_augment_transform()
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=trans)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    return train_set, test_set




def get_cifar100_dataloaders_sample(data_folder, k=4096, mode='exact', is_sample=True, percent=1.0, no_labels=False):
    """
    cifar 100
    """
    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent,
                                       no_labels=no_labels)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    
    return train_set, test_set
