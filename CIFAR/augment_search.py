import argparse
import logging
import os
import random
import sys
import time
import math
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

from dataset.cifar100 import get_cifar100_dataloaders, build_augmentation_transform, test_transform
from models.util import Centroid
from torchmetrics.aggregation import MeanMetric
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

    def clamp(x, low=0.0, high=1.0):
        return max(min(x, high), low)

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
    model_t = utils.load_teacher_model(model_t, args.teacher_path)
    model_t = model_t.eval()

    def populate_cmi():
        cmi = Centroid().to(device)
        train_loader, _ = get_loaders(test_transform) # First pass  
        with torch.no_grad():
            for (img, labels) in train_loader:
                img = img.to(device)
                labels = labels.to(device)
                pred = model_t(img)
                cmi(pred, labels)
        cmi.update_centroids()
        return cmi 
    
    cmi = populate_cmi()

        
    criterion = nn.CrossEntropyLoss()
    best_score = -10000.0

    def do_iteration(i, best_score, do_print=True, use_augs=None, mask_wrong=False):
        out = {}
        if use_augs is None:
            augs = generate_augments(i + args.seed)
            out["t1"] = augs[0].name
            out["t2"] = augs[1].name
            augs = build_augmentation_transform(augs)
            # augs = build_augmentation_transform([TAW()])
        else:
            augs = build_augmentation_transform(use_augs)

        train_loader, _ = get_loaders(augs) 
        avg_train_loss = MeanMetric().to(device)
        avg_cmi_loss = MeanMetric().to(device)
        avg_masked_cmi_loss = MeanMetric().to(device)
        total = 0
        correct_classified = 0
        with torch.no_grad():
            for (img, labels) in train_loader:
                img = img.to(device)
                labels = labels.to(device)
                pred = model_t(img)
                _, predicted = torch.max(pred.data, 1)
                # cmi(pred, labels)
                assert(cmi.is_ready)
                loss = criterion(pred, labels)
                avg_train_loss.update(loss)
                loss_cmi = cmi.get_loss(pred, labels)
                loss_masked_cmi = cmi.get_loss(pred[predicted == labels], labels[predicted == labels])
                # Experimental masking
                avg_cmi_loss.update(loss_cmi)
                avg_masked_cmi_loss.update(loss_masked_cmi)
                correct_classified += (predicted == labels).sum().item()
                if total == 0:
                    # If less than 50% accuracy on first minibatch, break
                    if correct_classified < 0.5 * float(img.size()[0]):
                        out.update({"idx": i, "loss_score": -10000, "cmi_score": -10000, "score": -10000, "masked_cmi": -10000})
                        return out
                total += pred.size(0)

        acc = correct_classified / total
        loss_score = -1.0 * avg_train_loss.compute().item()
        cmi_score = -1.0 * avg_cmi_loss.compute().item() 
        masked_cmi_score = -1.0 * avg_masked_cmi_loss.compute().item() 
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
                    "acc": acc, "masked_cmi": masked_cmi_score})
        return out

    scores = []
    best_score = -100.0

    for outer in range(0, 1000):
        temp = do_iteration(outer, best_score, do_print=True)
        if temp["score"] > best_score:
            best_score = temp["score"]
        scores.append(temp)

    scores.sort(key=lambda x: x["score"], reverse=True)
    import json
    with open("augment_scores3.json", "w") as f:
        # Dump the data into the file
        json.dump(scores, f)


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

    good = TAW()

    do_iteration(0, -10000, use_augs=[good])

if __name__ == "__main__":
    main()




