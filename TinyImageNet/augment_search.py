import argparse
import logging
import os
import random
import sys
import time
import math
import json
from torch.utils.data import DataLoader, Subset, Dataset

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import ConfusionMatrix
from tqdm.auto import tqdm
from torchvision.transforms import v2
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torchvision.transforms import v2
from torch import Tensor
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from models import model_dict
from data_loaders.imagenet_data_loader import imagenet_data_loader

class Settings():
    def __init__(self, transform_name):
        self.transform = transform_name
        self.workers = 8
        self.batch_size = 50

class Centroid(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        self.is_ready = False
        self.centroids = nn.Parameter(torch.zeros((num_classes, num_classes)), requires_grad=False)
        self.storage = nn.Parameter(torch.zeros((num_classes, num_classes)), requires_grad=False)
        self.count = nn.Parameter(torch.zeros(num_classes, dtype=torch.float), requires_grad=False)

    def forward(self, logits: torch.tensor, targets: torch.tensor):
        # storage [num_classes, num_classes] AKA [target, average logits]
        with torch.no_grad():
            output = torch.nn.functional.softmax(logits, 1)
            if targets.ndim == 1:
                self.count += torch.bincount(targets.flatten(), minlength=self.num_classes).float()
                if targets.dim() != output.dim():
                    targets = targets.expand((self.num_classes, -1)).T
                else:
                    raise NotImplementedError
                out = torch.zeros_like(self.storage).to(logits.device)
                out.scatter_reduce_(0, targets, output, "sum")
                self.storage += out
            else:
                self.count += targets.sum(dim=0)
                centroids = torch.einsum("bi,bj->ij", output, targets)
                self.storage += centroids
        

    def update_centroids(self):
        self.is_ready = True
        divide = self.count.expand((self.num_classes, -1)).T
        self.centroids.copy_(self.storage / divide)
        self.storage.copy_(torch.zeros_like(self.storage))
        self.count.copy_(torch.zeros_like(self.count))

    def get_centroids(self, target):
        return torch.index_select(self.centroids, 0, target)

    def get_loss(self, logits: torch.tensor, targets: torch.tensor):
        if self.is_ready:
            if targets.ndim > 1: 
                centroids = torch.einsum("bi,ij->bj", targets, self.centroids)
            else:
                centroids = self.get_centroids(targets)
            surrogate_loss = -1.0 * torch.nn.functional.kl_div(centroids.log(), 
                                                               torch.nn.functional.log_softmax(logits, 1), 
                                                               reduction="batchmean", log_target=True)
            return surrogate_loss
        else:
            return 0.0


def main():
    start_time = time.time()
    model_names = model_dict.keys()

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
    parser.add_argument('--teacher_arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--teacher_path', type=str)
    parser.add_argument('--seed', type=int, default=None, help='seed for initialization')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

    # training settings
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size for training')
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
    parser.add_argument('--pretrain_path', type=str, default='./results/ResNet20_CIFAR10/fp/checkpoint/last_checkpoint.pth', help='path for pretrained full-preicion model')

    args = parser.parse_args()
    arg_dict = vars(args)



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
    
    args.num_classes = 200

    # if args.dataset == "cifar10":
    #     args.num_classes = 10
    #     train_dataset, _ = get_cifar10_dataloaders("./dataset/data/CIFAR10/")
    #     train_dataset2, _ = get_cifar10_dataloaders("./dataset/data/CIFAR10/")
    # else:
    #     args.num_classes = 100
    #     train_dataset, _ = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/", is_instance=False)
    #     train_dataset2, _ = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/", is_instance=False)

    model_t = model_dict[args.teacher_arch](pretrained=False, num_classes=args.num_classes)
    # model_t = models.__dict__[args.teacher_arch](pretrained=False, num_classes=num_classes)
    if os.path.isfile(args.teacher_path):
        print("Loading checkpoint '{}'".format(args.teacher_path))
        checkpoint_t = torch.load(args.teacher_path, map_location = lambda storage, loc: storage.cuda(device))
        model_t.load_state_dict(checkpoint_t['state_dict'])
        print("Loaded, epoch: {}, acc: {})".format(checkpoint_t['epoch'], checkpoint_t['best_prec1']))
    # model_t = model_t.train()
    model_t = model_t.to(device)

    def get_loaders(augmentation, seed_offset=0, var=False): 
        _init_fn(seed_offset)
        train, _ = imagenet_data_loader(Settings(augmentation), multi=var)
        return train

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

    def populate_cmi(loader, use_cutmix=False):
        cmi = Centroid(num_classes=args.num_classes).to(device)
        cutmix = v2.CutMix(num_classes=args.num_classes)
        # train_loader, _ = get_loaders(test_transform) # First pass  
        with torch.no_grad():
            for (img, labels) in loader:
                img = img.to(device)
                labels = labels.to(device)
                if use_cutmix:
                    img, labels = cutmix(img, labels)
                pred = model_t(img)
                cmi(pred, labels)
        cmi.update_centroids()
        return cmi 
    
    class VarOpt():
        def __init__(self):
            self.prob = []
            self.entropy = []
            self.all_avg_prob = []
            self.epoch = 0
            self.total_step = 0

    
    def check_variance(opt, logit_t):
        # Check teacher's avg probability, Neurips'22
        p = torch.nn.functional.softmax(logit_t, dim=-1)
        opt.prob += [p] # [batch_size, num_classes]
        opt.entropy += [(-p * torch.log(p)).sum(dim=-1)] # [batch_size]
        # if opt.lw_mix is not None:
        #     p = F.softmax(input_mix_logit_t, dim=-1)
        #     opt.prob += [p] # [batch_size, num_classes]
        #     opt.entropy += [(-p * torch.log(p)).sum(dim=-1)] # [batch_size]
        # num = 5 if opt.lw_mix is None else 10 # 10 is empirically set
        num = 5
        if len(opt.prob) >= num: 
            prob = torch.cat(opt.prob, dim=0) # [..., num_classes]
            entropy = torch.cat(opt.entropy, dim=0)
            avg_prob = prob.mean(dim=0) # [num_classes]
            opt.all_avg_prob += [avg_prob]
            opt.prob = [] # reset 

            all_avg_prob = torch.stack(opt.all_avg_prob, dim=0) # [Num, num_classes]
            avg_prob_std = all_avg_prob.std(dim=0)
            # std_str = ' '.join(['%.6f' % x for x in tensor2list(avg_prob_std)])
            std_str = '%.6f' % avg_prob_std.mean().item()
            # print(f'Check T prob: NumOfSampledStd {len(opt.all_avg_prob)} Epoch {opt.epoch} Step {idx} TotalStep {opt.total_step} MeanStd {std_str} MeanEntropy {entropy.mean().item():.6f}')
            return std_str
        return None
    
    def compute_variance_loop(aug_loader, use_cutmix=False):
        NUM_EPOCHS = 1
        cutmix = v2.CutMix(num_classes=args.num_classes)
        opt = VarOpt()
        var_str = None
        # train_loader, _ = get_loaders(test_transform) # First pass
        for ep in range(NUM_EPOCHS):
            with torch.no_grad():
                opt.epoch = ep
                for (img, img2, labels) in aug_loader:
                    img = img.to(device)
                    labels = labels.to(device)
                    img2 = img2.to(device)
                    if use_cutmix:
                        img, labels1 = cutmix(img, labels)
                        img2, labels2 = cutmix(img2, labels)
                    pred = model_t(img)
                    pred2 = model_t(img2)
                    preds = torch.cat([pred, pred2], dim=0)
                    res = check_variance(opt, preds)
                    if res is not None:
                        var_str = res
                    opt.total_step += 1
        return var_str

    criterion = nn.CrossEntropyLoss()
    entropy_no_reduce = nn.CrossEntropyLoss(reduction="none")
    kl_div = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def do_iteration(i, augs, do_print=True, use_cutmix=False, name=""):
        # model_t.train()
        # reset_model(model_t)
        # dfs_freeze(model_t)
        # unfreeze(model_t.classifier)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_t.parameters()), lr=1e-2, amsgrad=True)
        out = {}
        out["t1"] = name
        if use_cutmix:
            out["t2"] = "CutMix"
        else:
            out["t2"] = "None"
        # augs = build_augmentation_transform(use_augs)
        train_loader = get_loaders(augs)
        var_loader = get_loaders(augs, var=True)
        avg_var = compute_variance_loop(var_loader, use_cutmix=use_cutmix)
        # train_loader2, _ = get_loaders(augs, seed_offset=1)
        avg_train_loss = MeanMetric().to(device)
        avg_min_loss = MeanMetric().to(device)
        avg_cmi_loss = MeanMetric().to(device)
        avg_masked_cmi_loss = MeanMetric().to(device)
        avg_vec_sim = MeanMetric().to(device)
        avg_logit_sim = MeanMetric().to(device)
        avg_entropy = MeanMetric().to(device)
        # conf = ConfusionMatrix(task="multiclass", num_classes=100).to(device)
        cutmix = v2.CutMix(num_classes=args.num_classes)
        total = 0
        correct_classified = 0
        cmi = populate_cmi(train_loader, use_cutmix=use_cutmix)
        centroid_div = kl_div(cmi.centroids.log(), torch.eye(args.num_classes, dtype=torch.float32).to(device))

        with torch.no_grad():
            for (img, labels) in train_loader:
                img = img.to(device)
                # img2 = img2.to(device)
                labels = labels.to(device)
                if use_cutmix:
                    img, labels = cutmix(img, labels)
                pred = model_t(img)
                # feat2, _, pred2 = model_t(img2, is_feat=True)

                _, predicted = torch.max(pred.data, 1)

                entropy1 = torch.distributions.Categorical(logits=pred).entropy()
                # entropy2 = torch.distributions.Categorical(logits=pred2).entropy()
                # print(logit_sim[0:10])
                avg_entropy.update(entropy1)
                # avg_entropy.update(entropy2)
                # cmi(pred, labels)
                assert(cmi.is_ready)
                ent = entropy_no_reduce(pred, labels)
                # ent2 = entropy_no_reduce(pred2, labels)
                # min_loss = torch.minimum(ent, ent2).mean()
                loss = criterion(pred, labels)
                # loss2 = criterion(pred2, labels)


                avg_train_loss.update(loss)

                loss_cmi = cmi.get_loss(pred, labels)
                # loss_masked_cmi = cmi.get_loss(pred[predicted == labels], labels[predicted == labels])
                # Experimental masking
                avg_cmi_loss.update(loss_cmi)
                # avg_masked_cmi_loss.update(loss_masked_cmi)
                if predicted.ndim == labels.ndim: 
                    correct_classified += (predicted == labels).sum().item()
                total += pred.size(0)

        acc = correct_classified / total
        loss_score = -1.0 * avg_train_loss.compute().item()
        min_loss_score = -1.0 * avg_min_loss.compute().item()
        cmi_score = -1.0 * avg_cmi_loss.compute().item() 
        masked_cmi_score = -1.0 * avg_masked_cmi_loss.compute().item() 
        # confusion = conf.compute().cpu().tolist()
        # score = 0.5 * loss_score + cmi_score
        score = loss_score + cmi_score

        if do_print:
            print(f"Configuration #{i}")
            print(f"Augmentations: {augs}" )
            print(f"Loss Score: {loss_score:.3f}")
            print(f"CMI Score: {cmi_score:.3f}")
            print(f"Centroid Div: {centroid_div.item():.3f}")
            print(f"Our Score: {(cmi_score - centroid_div.item()):.3f}")
            print(f"Variance Score: {avg_var}")

        out.update({"idx": i, "loss_score": loss_score, "cmi_score": cmi_score, "score": score, 
                    "acc": acc, "masked_cmi": masked_cmi_score, "min_loss": min_loss_score, 
                    "logit_sim": avg_logit_sim.compute().item(), "vector_sim": avg_vec_sim.compute().item(), 
                    "entropy": avg_entropy.compute().item(), "centroid_div": centroid_div.item(),
                    "centroids": cmi.centroids.tolist(), "var": avg_var})
        return out

    scores = []

    special = [
        ("AugMix", "augmix"),
        ("AutoAugmentCifar", "auto"),
        ("AutoAugmentImagenet", "autoimg"),
        ("AutoAugmentSVHN", "autosvhn"),
        ("Erasing", "erasing"),
        ("RandAugment", "rand"),
        ("TrivialAugment", "trivial"),
        ("None", "none"),
    ]

    idx = 0
    scores = []
    for (name, aug) in special:
        temp = do_iteration(idx, aug, do_print=True, use_cutmix=False, name=name)
        idx += 1
        scores.append(temp)
        temp2 = do_iteration(idx, aug, do_print=True, use_cutmix=True, name=name)
        idx += 1
        scores.append(temp2)

    # temp = do_iteration(0, -10000, use_augs=[auto], do_print=True, use_cutmix=True)
    # temp2 = do_iteration(1, -10000, use_augs=[TAW()], do_print=True, use_cutmix=True)
    # temp3 = do_iteration(2, -10000, use_augs=[auto], do_print=True, use_cutmix=False)
    # temp4 = do_iteration(3, -10000, use_augs=[TAW()], do_print=True, use_cutmix=False)
    # scores = [temp, temp2, temp3, temp4]
    arch = args.teacher_arch.split("_")[0]
    with open(f"known_{arch}_{args.num_classes}.json", "w") as f:
        # Dump the data into the file
        json.dump(scores, f)


if __name__ == "__main__":
    main()