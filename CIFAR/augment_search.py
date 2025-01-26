import argparse
import logging
import os
import random
import sys
import time
import itertools
from torch.utils.data import DataLoader, Subset, Dataset
from dataclasses import dataclass
import torchvision

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from safetensors.torch import save_file

import utils

from models.custom_models_resnet import *
from models.custom_models_vgg import * 

from dataset.cifar100 import get_cifar100_dataloaders, build_augmentation_transform
from models.util import Centroid
from torchmetrics.aggregation import MeanMetric
from tqdm.auto import tqdm
from torchvision.transforms import v2


start_time = time.time()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (CIFAR)")

# data and model
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--teacher_arch', type=str, default='resnet20_fp', help='model architecture')
parser.add_argument('--teacher_path', type=str)
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

# training settings
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--optimizer_m', type=str, default='SGD', choices=('SGD','Adam'), help='optimizer for model paramters')
parser.add_argument('--lr_m', type=float, default=1e-1, help='learning rate for model parameters')
parser.add_argument('--lr_m_end', type=float, default=0.0, help='final learning rate for model parameters (for cosine)')
parser.add_argument('--decay_schedule_m', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for model parameters')
parser.add_argument('--lr_scheduler_m', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')
parser.add_argument('--use_cmi', type=str2bool, default=False, help="Track CMI")
parser.add_argument('--cmi_weight', type=float, default=0.0, help="Contextual Mutual Information weight")


# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
parser.add_argument('--log_dir', type=str, default='./results/ResNet20_CIFAR10/fp/')
parser.add_argument('--load_pretrain', type=str2bool, default=False, help='load pretrained full-precision model')
parser.add_argument('--pretrain_path', type=str, default='./results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth', help='path for pretrained full-preicion model')

args = parser.parse_args()
arg_dict = vars(args)


### make log directory
# if not os.path.exists(args.log_dir):
#     os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

# logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
#                     level=logging.INFO,
#                     format='')
# log_string = 'configs\n'
# for k, v in arg_dict.items():
#     log_string += "{}: {}\t".format(k,v)
#     print("{}: {}".format(k,v), end='\t')
# logging.info(log_string+'\n')
# print('')

### GPU setting
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device(f"cuda:{args.gpu_id}")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### set the seed number
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

args.num_classes = 100 # Todo not hardcode

train_dataset, test_dataset = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/", is_instance=False)




def get_loaders(augmentations): 
    train_ds = Subset(train_dataset, indices=torch.arange(50000))
    train_ds.dataset.transform = augmentations
    train_loader = DataLoader(dataset=train_ds,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            worker_init_fn=None if args.seed is None else _init_fn)
    test_loader = DataLoader(dataset=test_dataset,
                                            batch_size=100,
                                            shuffle=False,
                                            num_workers=args.num_workers)
    return train_loader, test_loader 

from torchvision.transforms import v2
# transforms = v2.Compose([
#     v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
#     v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
#     # ...
#     v2.RandomResizedCrop(size=(224, 224), antialias=True),  # Or Resize(antialias=True)
#     # ...
#     v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
#     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

def clamp(x, low=0.0, high=1.0):
    return max(min(x, high), low)


augmentations = [
    lambda a, b: v2.ColorJitter(brightness=clamp(a), saturation=clamp(b)),
    lambda a, _: v2.RandomInvert(p=clamp(a)),
    lambda a, b: v2.RandomAdjustSharpness(clamp(a), p=clamp(b)),
    lambda a, b: v2.RandomErasing(p=clamp(a), scale=(0.02, clamp(b, 0.03))),
    lambda a, _: v2.RandomEqualize(p=clamp(a)),
    lambda a, b: v2.RandomSolarize(clamp(a), p=clamp(b)),
    lambda a, _: v2.GaussianBlur(3, sigma=(0.1, clamp(a, 0.1, 5.0))),
    lambda a, b: v2.RandomPosterize(int(clamp(10 * a, 1.0, 8.0)), p=clamp(b)),
    lambda a, _: v2.RandomAutocontrast(p=clamp(a)),
    lambda a, b: v2.RandomAffine(degrees=0, translate=[clamp(a), clamp(a)], shear=[clamp(b), clamp(b)]), # This may be more complicated 
]

# train_loader, test_loader = get_loaders(

### initialize model
# model_class = globals().get(args.arch)
# model = model_class(args)

# import copy
# foo = copy.deepcopy(model)
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device(f"cuda:{args.gpu_id}")

### set the seed number
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

model_class_t = globals().get(args.teacher_arch)
model_t = model_class_t(args)
model_t.to(device)
baseline_cmi = None
cmi = Centroid().to(device)

model_t = utils.load_teacher_model(model_t, args.teacher_path)
model_t = model_t.eval()

@dataclass 
class Augmentation:
    idx: int
    priority: float
    x1: float
    x2: float 

    def get_aug(self):
        return augmentations[self.idx](self.x1, self.x2)

    
criterion = nn.CrossEntropyLoss()
root_seed = np.random.SeedSequence()
best_score = -10000.0

foo = torchvision.datasets.CIFAR100(root="./dataset/data/CIFAR100/",
                download=True,
                train=False,
                transform=None)

print(np.asarray(foo[0][0]))
assert(False)


def do_iteration(i, best_score):
    gen = np.random.Generator(np.random.PCG64(seed=args.seed + i))
    # aug_seed = np.random.Generator((len(augmentations) * 3))
    aug_seed = gen.random((len(augmentations) * 3))
    sub = [Augmentation(i, arr[0], arr[1], arr[2]) for i, arr in enumerate(np.array_split(aug_seed, len(augmentations)))]
    sub = sorted(sub, key=lambda a: a.priority, reverse=True)
    take_k = 5
    sub = sub[0:take_k]
    # print(aug_seed)
    augs = build_augmentation_transform([a.get_aug() for a in sub])
    # print(augs)
    train_loader, test_loader = get_loaders(augs) 
    avg_train_loss = MeanMetric().to(device)
    avg_cmi_loss = MeanMetric().to(device)
    total = 0
    correct_classified = 0
    for ep in range(2):
        with torch.no_grad():
            for (img, labels) in train_loader:
                img = img.to(device)
                labels = labels.to(device)
                pred = model_t(img)
                loss = criterion(pred, labels)
                avg_train_loss.update(loss)
                cmi(pred, labels) # Update for next epoch
                if ep != 0:
                    assert(cmi.is_ready)
                    loss_cmi = cmi.get_loss(pred, labels)
                    avg_cmi_loss.update(loss_cmi)
                _, predicted = torch.max(pred.data, 1)
                correct_classified += (predicted == labels).sum().item()
                if total == 0:
                    # If less than 80% accuracy on first minibatch, break
                    if correct_classified < 0.8 * float(img.size()[0]):
                        return best_score

                total += pred.size(0)

            cmi.update_centroids()
    loss_score = -1.0 * avg_train_loss.compute().item()
    cmi_score = -1.0 * avg_cmi_loss.compute().item() 
    score = loss_score + cmi_score
    if score > best_score:
        best_score = score
        print(f"Configuration #{i}")
        print(f"Augmentations: {augs}" )
        print(f"Accuracy: {correct_classified / total:.3f}")
        # Want to maximize both
        print(f"Loss Score: {loss_score:.3f}")
        print(f"CMI Score: {cmi_score:.3f}")
    else:
        print(f"Configuration #{i} did not pass")

    return best_score

for i in range(500):
    best_score = do_iteration(i, best_score)





