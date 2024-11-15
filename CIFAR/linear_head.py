import argparse
import logging
import os
import random
import sys
import time 
import copy

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn

from models.custom_modules import *
from models.custom_models_resnet import *
from models.custom_models_vgg import *

from utils import *
from utils import printRed

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders


print(f"Cuda: {torch.cuda.is_available()}")


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
parser.add_argument('--arch', type=str, default='resnet20_quant', help='model architecture')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')

# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
parser.add_argument('--log_dir', type=str, default='./results/ResNet20_CIFAR10/W1A1/')
parser.add_argument('--load_pretrain', type=str2bool, default=False, help='load pretrained full-precision model')
parser.add_argument('--pretrain_path', type=str, default='./results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth', help='path for pretrained full-preicion model')

args = parser.parse_args()
arg_dict = vars(args)

### make log directory
if not os.path.exists(args.log_dir):
    os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')
log_string = 'configs\n'
for k, v in arg_dict.items():
    log_string += "{}: {}\t".format(k,v)
    print("{}: {}".format(k,v), end='\t')
logging.info(log_string+'\n')
print('')

### GPU setting
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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



if args.dataset == 'cifar10':
    args.num_classes = 10
    train_dataset, test_dataset = get_cifar10_dataloaders(data_folder="./dataset/data/CIFAR10/")

elif args.dataset == 'cifar100':
    args.num_classes = 100
    train_dataset, test_dataset = get_cifar100_dataloaders(data_folder="../data/CIFAR100/", is_instance=False)

else:
    raise NotImplementedError


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           worker_init_fn=None if args.seed is None else _init_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False,
                                          num_workers=args.num_workers)
printRed(f"dataset: {args.dataset}, num of training data (50,000): {len(train_dataset)}, number of testing data (10,000): {len(test_dataset)}")                                          


### initialize model
model_class = globals().get(args.arch)
model = model_class(args)
model.to(device)


num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))

print(f"Cuda: {torch.cuda.is_available()}")

def remove_indirection(name):
    if "f." in name:
        return name.replace("f.", "")
    return name

if args.load_pretrain:
    trained_model = torch.load(args.pretrain_path)
    current_dict = model.state_dict()
    printRed("Pretrained full precision weights are initialized")
    logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''
    print(list(current_dict.keys()))
    print(list(trained_model.keys()))
    if "model" not in trained_model:
        for key in trained_model.keys():
            new_key = remove_indirection(key)
            if new_key in current_dict.keys():
                # print(key)
                log_string += '{}\t'.format(new_key)
                current_dict[new_key].copy_(trained_model[key])
            else:
                print(f"Could not find: {key}")
        for key in current_dict.keys():
            new_key = "f." + key
            if new_key not in trained_model.keys():
                print(f"Could not set {new_key}")
    else:
        for key in trained_model['model'].keys():
            if key in current_dict.keys():
                # print(key)
                log_string += '{}\t'.format(key)
                current_dict[key].copy_(trained_model['model'][key])
    logging.info(log_string+'\n')
    model.load_state_dict(current_dict)
    # For testing accuracy
    # pretrained_test_acc = trained_model['test_acc']
    # pretrained_epoch = trained_model['epoch']
    # print(f"The test accuracy of the pretrained model is: {pretrained_test_acc}, from epoch: {pretrained_epoch}")
else:
    printRed("Not initialized by the pretrained full precision weights")

class FlatNormal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        feature = torch.flatten(x, start_dim=1)
        return nn.functional.normalize(feature, dim=-1)


if "model" not in trained_model:
    model.classifier = FlatNormal()
    # model.classifier = nn.Identity()
else:
    model.classifier = nn.Identity()

model = model.to(device)

for batch in test_loader:
    model = model.eval()
    data = batch[0].cuda()
    data[:] = 1.0
    vals = model(data)
    print(vals.mean())
    break

from sklearn.linear_model import LogisticRegression

train_set = []
train_labels = []
test_set = []
test_labels = []
model = model.eval()
with torch.no_grad():
    for batch in train_loader:
        x, y = batch
        x = x.to(device)
        pred = model(x)
        train_set.append(pred.cpu())
        train_labels.append(y)

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        pred = model(x)
        test_set.append(pred.cpu())
        test_labels.append(y)

    train_set = torch.cat(train_set)
    train_labels = torch.cat(train_labels)
    test_set = torch.cat(test_set)
    test_labels = torch.cat(test_labels)

print(train_set.size())
print(train_labels.size())
print(test_set.size())
print(test_labels.size())

regr = LogisticRegression(penalty=None, solver="lbfgs")
regr.fit(train_set, train_labels)
mean = regr.score(test_set, test_labels) # 25.5%, 26.2%
print(f"Accuracy: {mean}")
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
pca = PCA(n_components=50)
test_transform = pca.fit_transform(test_set)
tsne = TSNE(2)
test_transform = tsne.fit_transform(test_transform)

f, ax = plt.subplots()
rand_cls = [1, 2, 17, 50, 99]
for cls in rand_cls:
    idx = test_labels == cls
    data = test_transform[idx]
    ax.scatter(data[:, 0], data[:, 1], label=cls)
f.legend()
f.savefig("fig.png", transparent=False)
