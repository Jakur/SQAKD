from __future__ import print_function

import os
import socket
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
import sys

from torchvision import datasets, transforms
from torchvision.transforms import v2

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

def found_augment_transform():
    from torchvision.transforms import v2
    aug = v2.Compose([
    v2.ColorJitter(brightness=(0.4955052744829549, 1.504494725517045), saturation=(0.019858587004238704, 1.9801414129957613)),
    v2.RandomInvert(p=0.05811370103932734),
    v2.RandomEqualize(p=0.10200932855456979),
    v2.RandomPosterize(p=0.9060121628537261, bits=5),
    v2.RandomAffine(degrees=[0.0, 0.0], translate=[0.25175223663172586, 0.25175223663172586], shear=[0.4597930748859477, 0.4597930748859477])
    ])
    # aug = v2.Compose([
    # v2.RandomInvert(p=0.1217276231215455),
    # v2.RandomAdjustSharpness(p=0.5129257814705915, sharpness_factor=0.2991414325015239),
    # v2.RandomSolarize(p=0.2902056243045372, threshold=0.37053888043208205),
    # v2.RandomAutocontrast(p=0.916472038114996),
    # v2.ColorJitter(brightness=(0.10260029398324955, 1.8973997060167505), saturation=(0.02123074420087767, 1.9787692557991223))
    # ])
    # aug = v2.Compose([
    #     v2.RandomAdjustSharpness(p=0.37674046326081456, sharpness_factor=0.473388021620039),
    #     v2.ColorJitter(brightness=(0.7787606929447987, 1.2212393070552015), saturation=(0.14031598835343717, 1.859684011646563)),
    #     v2.GaussianBlur(kernel_size=(3, 3), sigma=[0.1, 0.24801515642962435]),
    #     v2.RandomAffine(degrees=[0.0, 0.0], translate=[0.05638597202397311, 0.05638597202397311], shear=[0.317924972803986, 0.317924972803986], fill=0),
    #     v2.RandomAutocontrast(p=0.7480985007308912),
    # ])
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        aug,
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

def build_augmentation_transform(extra: list):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.Compose(extra),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

def get_heavy_augment_transform():
    auto = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        auto,
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

def get_general_transform(trans):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        trans,
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

def get_augmix_transform():
    auto = v2.AugMix()
    return get_general_transform(auto)

def get_trivial_transform():
    trivial = transforms.TrivialAugmentWide()
    return get_general_transform(trivial)

def get_rand_transform():
    rand = v2.RandAugment()
    return get_general_transform(rand)

def get_erasing_transform():
    erasing = v2.RandomErasing()
    return get_general_transform(erasing)

def get_imagenet_aa_transform():
    aa = v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)
    return get_general_transform(aa)

def get_svhn_aa_transform():
    aa = v2.AutoAugment(v2.AutoAugmentPolicy.SVHN)
    return get_general_transform(aa)


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
        # imgs = [train_transform(base_img)]
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
    

def get_custom(custom):
    if isinstance(custom, str):
        if custom == "auto":
            trans = get_heavy_augment_transform()
        elif custom == "trivial":
            trans = get_trivial_transform()
        elif custom == "custom":
            trans = found_augment_transform()
        elif custom == "augmix":
            trans = get_augmix_transform()
        elif custom == "rand":
            trans = get_rand_transform()
        elif custom == "erasing":
            trans = get_erasing_transform()
        elif custom == "autoimg":
            trans = get_imagenet_aa_transform()
        elif custom == "autosvhn":
            trans = get_svhn_aa_transform()
        elif custom == "none":
            trans = train_transform
        else:
            return NotImplementedError
    else:
        trans = build_augmentation_transform(custom)
    return trans

def get_cifar100_dataloaders(data_folder, is_instance=False, self_supervised=False, is_augment=False, num_transforms=3, agg_trans=False, 
                             custom_transform=None, size=50000):
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
        if custom_transform is not None:
            trans = get_custom(custom_transform)
        else:
            trans = get_heavy_augment_transform()
        train_set = CIFAR100Augment(num_transforms=num_transforms, 
                                    root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=trans)
    else:
        if custom_transform is not None:
            trans = get_custom(custom_transform)
        elif agg_trans:
            trans = get_heavy_augment_transform()
        else:
            trans = train_transform

        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=trans)
        
    if size != 50000:
        train_set = Subset(train_set, indices=torch.arange(0, size))

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
