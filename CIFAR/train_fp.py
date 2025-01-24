import argparse
import logging
import os
import random
import sys
import time
import itertools

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from safetensors.torch import save_file

import utils

from models.custom_models_resnet import *
from models.custom_models_vgg import * 

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from models.util import Centroid, Embed
from torchmetrics.aggregation import MeanMetric
from linear_head import train_linear
from tqdm.auto import tqdm


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
parser.add_argument('--arch', type=str, default='resnet20_fp', help='model architecture')
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
parser.add_argument('--self_supervised', type=str2bool, default=False)
parser.add_argument('--aggressive_transforms', type=str2bool, default=False, help="Aggressive transforms")


# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
parser.add_argument('--log_dir', type=str, default='./results/ResNet20_CIFAR10/fp/')
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


if args.dataset == 'cifar10':
    args.num_classes = 10
    train_dataset, test_dataset = get_cifar10_dataloaders(data_folder="./dataset/data/CIFAR10/")

elif args.dataset == 'cifar100':
    args.num_classes = 100
    train_dataset, test_dataset = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/", is_instance=False, 
                                                           self_supervised=args.self_supervised, agg_trans=args.aggressive_transforms)

else:
    raise NotImplementedError

def get_loaders(train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            worker_init_fn=None if args.seed is None else _init_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=100,
                                            shuffle=False,
                                            num_workers=args.num_workers)
    return train_loader, test_loader 

train_loader, test_loader = get_loaders(train_dataset, test_dataset)

### initialize model
# model_class = globals().get(args.arch)
# model = model_class(args)

# import copy
# foo = copy.deepcopy(model)
from torchvision import models

# resnet = models.resnet50(pretrained=True)


### initialize optimizer, scheduler, loss function
# optimizer for model params
if args.self_supervised:
    from byol_pytorch import BYOL
    # model_class = globals().get(args.arch)
    # model = model_class(args)
    model = models.resnet50()

    for param in model.parameters():
        print(param.size())
    byol = BYOL(model, 32, hidden_layer=-2, projection_size=512, use_momentum=True).to(device)
    params = byol.parameters()
else:
    print(args.arch)
    model_class = globals().get(args.arch)
    print(model_class)
    model = model_class(args)
    model.to(device)
    params = model.parameters()


num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))

def get_optimizer(params):
    if args.optimizer_m == 'SGD':
        return torch.optim.SGD(params, lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer_m == 'Adam':
        return torch.optim.Adam(params, lr=args.lr_m, weight_decay=args.weight_decay)

optimizer_m = get_optimizer(params)
    
# scheduler for model params
if args.lr_scheduler_m == "step":
    if args.decay_schedule_m is not None:
        milestones_m = list(map(lambda x: int(x), args.decay_schedule_m.split('-')))
    else:
        milestones_m = [args.epochs+1]
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones_m, gamma=args.gamma)
elif args.lr_scheduler_m == "cosine":
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(args.log_dir)
total_iter = 0
# assert(False) # Dummy return
if args.self_supervised:
    best_acc = 0.0
    for ep in range(args.epochs):
        avg_train_loss = MeanMetric().to(device)
        avg_variance_div = MeanMetric().to(device)
        info_str = f"Starting self supervised epoch {ep}..."

        print(info_str)
        logging.info(info_str)
        byol.train()
        writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], ep)
        for (img, _) in tqdm(train_loader):
            img = img.to(device)

            optimizer_m.zero_grad()

            loss, variance_loss = byol(img)
            avg_train_loss.update(loss)
            avg_variance_div.update(variance_loss)
            writer.add_scalar("train/self_super_loss", loss.item(), total_iter)
            loss.backward()
            optimizer_m.step()
            byol.update_moving_average()
            total_iter += 1
        print(f"Average Train Loss: {avg_train_loss.compute().item()} / Average Variance: {avg_variance_div.compute().item()}")
        logging.info(f"Average Train Loss: {avg_train_loss.compute().item()}")

        scheduler_m.step()

        with torch.no_grad():
            if ep % 25 == 0 or ep == args.epochs - 1:
                byol.eval()
                model.classifier = nn.Identity()
                test_acc = 100.0 * train_linear(model, train_loader, test_loader, device)
                writer.add_scalar('test/acc', test_acc, ep)
                print("Current epoch: {:03d}\t Test accuracy: {}%".format(ep, test_acc))
                logging.info("Current epoch: {:03d}\t Test accuracy: {}%".format(ep, test_acc))
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save({
                        'epoch':ep,
                        'test_acc': test_acc,
                        'model':model.state_dict(),
                        'optimizer_m':optimizer_m.state_dict(),
                        'scheduler_m':scheduler_m.state_dict(),
                        'criterion':criterion.state_dict()
                    }, os.path.join(args.log_dir,f'checkpoint/best_checkpoint.pth'))
                if ep == args.epochs - 1:
                    torch.save({
                        'epoch':ep,
                        'test_acc': test_acc,
                        'model':model.state_dict(),
                        'optimizer_m':optimizer_m.state_dict(),
                        'scheduler_m':scheduler_m.state_dict(),
                        'criterion':criterion.state_dict()
                    }, os.path.join(args.log_dir,f'checkpoint/last_checkpoint.pth'))

    exit("Done")
    train_epochs = 50
    optimizer_m = torch.optim.SGD(model.classifier.parameters(), lr=1e-2)
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=train_epochs, eta_min=args.lr_m_end)
else:
    train_epochs = args.epochs

### train
best_acc = 0
baseline_cmi = None
cmi = Centroid().to(device)
for ep in range(train_epochs):
    if args.self_supervised:
        model.eval() # Turn off batch norm "learning"
    else:
        model.train()
    writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], ep)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer_m.zero_grad()
            
        pred = model(images)

        loss_t = criterion(pred, labels)
        if args.use_cmi and cmi.is_ready:
            loss_cmi = cmi.get_loss(pred, labels)
            writer.add_scalar("train/cmi", -1.0 * loss_cmi, total_iter)
            loss = loss_t + args.cmi_weight * loss_cmi.clamp(baseline_cmi, 0.0)
        else:
            loss = loss_t
        loss.backward()
        
        optimizer_m.step()
        writer.add_scalar('train/loss', loss.item(), total_iter)
        total_iter += 1
    
    scheduler_m.step()

    with torch.no_grad():
        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
            cmi(pred, labels)
            
        cmi.update_centroids()
        if baseline_cmi is None:
            mean_cmi = MeanMetric().to(device)
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                loss_cmi = cmi.get_loss(pred, labels)
                mean_cmi.update(loss_cmi)
                baseline_cmi = mean_cmi.compute().item()
        # tensors = {
        #     "centroids": cmi.centroids,
        #     "storage": cmi.storage,
        # }
        # save_file(tensors, "centroids.safetensors")
        # assert(False)
        test_acc = correct_classified/total*100
        writer.add_scalar('train/acc', test_acc, ep)

        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")
        logging.info("Current epoch: {:03d}\t Test accuracy: {}%".format(ep, test_acc))
        writer.add_scalar('test/acc', test_acc, ep)
        if args.epochs - ep < 25: # Only save near the end of the run
            torch.save({
                'epoch':ep,
                'test_acc': test_acc,
                'model':model.state_dict(),
                'optimizer_m':optimizer_m.state_dict(),
                'scheduler_m':scheduler_m.state_dict(),
                'criterion':criterion.state_dict()
            }, os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch':ep,
                'test_acc': test_acc,
                'model':model.state_dict(),
                'optimizer_m':optimizer_m.state_dict(),
                'scheduler_m':scheduler_m.state_dict(),
                'criterion':criterion.state_dict()
            }, os.path.join(args.log_dir,'checkpoint/best_checkpoint.pth'))  
    

checkpoint_path_last = os.path.join(args.log_dir, 'checkpoint/last_checkpoint.pth')
checkpoint_path_best = os.path.join(args.log_dir, 'checkpoint/best_checkpoint.pth')
utils.test_accuracy(checkpoint_path_last, model, logging, device, test_loader)
utils.test_accuracy(checkpoint_path_best, model, logging, device, test_loader)


print(f"Total time: {(time.time()-start_time)/3600}h")
logging.info(f"Total time: {(time.time()-start_time)/3600}h")

print(f"Save to {args.log_dir}")
logging.info(f"Save to {args.log_dir}")