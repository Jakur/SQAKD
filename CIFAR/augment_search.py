import argparse
import logging
import os
import random
import sys
import time
import math
import json
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

from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders, build_augmentation_transform, test_transform
from models.util import Centroid
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import ConfusionMatrix
from tqdm.auto import tqdm
from torchvision.transforms import v2
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torchvision.transforms import v2
from torch import Tensor
from torchvision.transforms import InterpolationMode, functional as F


def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class SingleAugment(torch.nn.Module):
    def __init__(self, transform_name: str, max_bucket: int, num_buckets=31):
        super().__init__()
        self.name = transform_name 
        self.max_bucket = max_bucket 
        self.num_buckets = num_buckets

    def forward(self, img: Tensor) -> Tensor:
        op_meta = TAW._augmentation_space(self.num_buckets)
        magnitudes, signed = op_meta[self.name]
        magnitude = (
            float(magnitudes[torch.randint(self.max_bucket, (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, self.name, magnitude, interpolation=InterpolationMode.NEAREST, fill=None)
    
    def __repr__(self):
        s = (
            f"{self.__class__.__name__}("
            f"transform={self.name}"
            f", max_mag={self.max_bucket}"
            f")"
        )
        return s

    


class TAW(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
    
    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels = 3
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = TAW._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

all_augs = sorted(list(TAW._augmentation_space(31).keys()))

def generate_augments(seed: int):
    gen = np.random.Generator(np.random.PCG64(seed=seed))
    augs = []
    for _ in range(2):
        op_idx = gen.integers(0, len(all_augs))
        max_mag = gen.integers(1, 7)
        aug = SingleAugment(all_augs[op_idx], max_mag, num_buckets=7)
        augs.append(aug)
    # augs = build_augmentation_transform(augs)
    return augs

def build_from_seeds(data: list):
    good = []
    for idx in data:
        candidate = generate_augments(idx)
        good.append(v2.Compose(candidate))
    good = v2.RandomChoice(good)
    return [good]

def main():
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
    parser.add_argument('--log_dir', type=str, default='./results/ResNet20_CIFAR10/fp/')
    parser.add_argument('--load_pretrain', type=str2bool, default=False, help='load pretrained full-precision model')
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

    if args.dataset == "cifar10":
        args.num_classes = 10
        train_dataset, _ = get_cifar10_dataloaders("./dataset/data/CIFAR10/")
        train_dataset2, _ = get_cifar10_dataloaders("./dataset/data/CIFAR10/")
    else:
        args.num_classes = 100
        train_dataset, _ = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/", is_instance=False)
        train_dataset2, _ = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/", is_instance=False)

    model_class_t = globals().get(args.teacher_arch)
    model_t = model_class_t(args)
    model_t.to(device)
    model_t = utils.load_teacher_model(model_t, args.teacher_path)
    model_t = model_t.eval()
    # model_t = model_t.train()

    # end = model_t.classifier.weight.clone().detach()
    # end_b = model_t.classifier.bias.clone().detach()

    def reset_model(model):
        model.classifier.load_state_dict({"bias": end_b, "weight": end})
        return model

    def dfs_freeze(model):
        for _, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            dfs_freeze(child)

    def unfreeze(layer):
        for name, child in layer.named_modules():
            for param in child.parameters():
                param.requires_grad = True

    def get_loaders(augmentations, seed_offset=0): 
        indices = torch.arange(50000)
        _init_fn(seed_offset)
        # train_indices = torch.arange(0, 40000)
        # val_indices = torch.arange(40000, 50000)
        train_ds = Subset(train_dataset, indices=indices)
        val_ds = Subset(train_dataset2, indices=indices)
        train_ds.dataset.transform = augmentations
        train_loader = DataLoader(dataset=train_ds,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_ds,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers)
        return train_loader, val_loader 

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
        
    criterion = nn.CrossEntropyLoss()
    entropy_no_reduce = nn.CrossEntropyLoss(reduction="none")
    kl_div = nn.KLDivLoss(reduction="batchmean", log_target=False)
    log_soft = nn.LogSoftmax(dim=1)
    cosine_sim = nn.CosineSimilarity(dim=1)
    best_score = -10000.0
    pool = nn.AdaptiveMaxPool2d((1, 1))

    # _, val_loader = get_loaders([]) 

    # vanilla_logits = []
    # model_t.eval()
    # avg = MeanMetric().to(device)
    # soft = nn.Softmax(dim=1)
    # with torch.no_grad():
    #     for (img, labels) in val_loader:
    #         img = img.to(device)
    #         labels = labels.to(device)
    #         pred = model_t(img)
    #         loss = criterion(pred, labels)
    #         avg.update(loss)
    #         vanilla_logits.append(soft(pred).cpu())
    
    # print(f"Average Baseline: {avg.compute().item()}")
    # assert(False)
    # vanilla_logits = torch.cat(vanilla_logits, dim=0)

    def do_iteration(i, best_score, do_print=True, use_augs=None, mask_wrong=False, use_cutmix=False, name=""):
        # model_t.train()
        # reset_model(model_t)
        # dfs_freeze(model_t)
        # unfreeze(model_t.classifier)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_t.parameters()), lr=1e-2, amsgrad=True)
        out = {}
        if use_augs is None:
            augs = generate_augments(i + args.seed)
            out["t1"] = augs[0].name
            out["t2"] = augs[1].name
            augs = build_augmentation_transform(augs)
            # augs = build_augmentation_transform([TAW()])
        else:
            out["t1"] = name
            if use_cutmix:
                out["t2"] = "CutMix"
            else:
                out["t2"] = "None"
            augs = build_augmentation_transform(use_augs)

        train_loader, _ = get_loaders(augs) 
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
                feat, _, pred = model_t(img, is_feat=True)
                # feat2, _, pred2 = model_t(img2, is_feat=True)

                _, predicted = torch.max(pred.data, 1)

                # for f in feat:
                #     print(f.size())
                # assert(False)
                # for f1, f2 in zip(feat, feat2):
                #     if f1.ndim > 2:
                #         f1 = pool(f1).view(args.batch_size, -1)
                #         f2 = pool(f2).view(args.batch_size, -1)
                #     print(f1.size())
                #     vec_sim = -kl_div(f1, f2)
                #     print(vec_sim)
                    # print(vec_sim[0:10])

                # vec_sim = cosine_sim(feat[-1], feat2[-1])
                # logit_sim = cosine_sim(pred, pred2)
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
                # avg_train_loss.update(loss2)
                # avg_min_loss.update(min_loss)
                # avg_vec_sim.update(vec_sim)
                # avg_logit_sim.update(logit_sim)
                # conf.update(pred, labels)
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

        if score > best_score:
            best_score = score
            if do_print:
                print(f"Configuration #{i}")
                print(f"Augmentations: {augs}" )
                print(f"Accuracy: {acc:.3f}")
                # Want to maximize both
                print(f"Loss Score: {loss_score:.3f}")
                print(f"CMI Score: {cmi_score:.3f}")
        elif do_print:
            print(f"Configuration #{i} did not pass")

        out.update({"idx": i, "loss_score": loss_score, "cmi_score": cmi_score, "score": score, 
                    "acc": acc, "masked_cmi": masked_cmi_score, "min_loss": min_loss_score, 
                    "logit_sim": avg_logit_sim.compute().item(), "vector_sim": avg_vec_sim.compute().item(), 
                    "entropy": avg_entropy.compute().item(), "centroid_div": centroid_div.item(),
                    "centroids": cmi.centroids.tolist()})
        return out

    scores = []
    best_score = -100.0
    # use_cutmix = True

    # for outer in range(0, 500):
    #     temp = do_iteration(outer, best_score, do_print=True, use_cutmix=use_cutmix)
    #     if temp["score"] > best_score:
    #         best_score = temp["score"]
    #     scores.append(temp)

    # scores.sort(key=lambda x: x["score"], reverse=True)
    # with open("augment_scores_resnet.json", "w") as f:
    #     # Dump the data into the file
    #     json.dump(scores, f)


    # good = []
    # for d in scores[0:5]:
    #     score = d["score"]
    #     seed = d["idx"]
    #     print(f"Score: {score}, Seed {seed}")
    #     candidate = generate_augments(d["idx"] + args.seed)
    #     print(candidate)
    #     good.append(v2.Compose(candidate))

    # good = [v2.Compose(generate_augments(x + args.seed)) for x in [391, 568, 692, 721, 270, 734, 918, 362, 903, 170]]

    # good = v2.RandomChoice(good)

    special = [
        ("AugMix", v2.AugMix()),
        ("AutoAugmentCifar", v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10)),
        ("AutoAugmentImagenet", v2.AutoAugment(policy=v2.AutoAugmentPolicy.IMAGENET)),
        ("AutoAugmentSVHN", v2.AutoAugment(policy=v2.AutoAugmentPolicy.SVHN)),
        ("Erasing", v2.RandomErasing()),
        ("RandAugment", v2.RandAugment()),
        ("TrivialAugment", TAW()),
        ("None", v2.Identity()),
    ]
    # good = TAW()
    idx = 0
    scores = []
    for (name, aug) in special:
        temp = do_iteration(idx, -10000, use_augs=[aug], do_print=True, use_cutmix=False, name=name)
        idx += 1
        scores.append(temp)
        temp2 = do_iteration(idx, -10000, use_augs=[aug], do_print=True, use_cutmix=True, name=name)
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




