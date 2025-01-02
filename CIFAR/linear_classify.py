import os
from pathlib import Path
from typing import Dict

import torch
import pytorch_lightning as light
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Module, CrossEntropyLoss, Linear
from torch.utils.data import DataLoader 
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import LinearClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero

from typing import Any, Dict, List, Tuple, Union

from pytorch_lightning import LightningModule
from torch import Tensor

from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.scheduler import CosineWarmupScheduler


class MyLinearClassifier(LinearClassifier):

    def forward(self, images: Tensor) -> Tensor:
        if self.freeze_model:
            with torch.no_grad():
                features = self.model.backbone(images).squeeze()
                # print(features.size())
                features = torch.nn.functional.normalize(features, dim=1)
                # print(features.size())
        else:
            features = self.model.backbone(images).squeeze()
            features = torch.nn.functional.normalize(features, dim=1)
        output: Tensor = self.classification_head(features)
        return output
    # def configure_optimizers(  # type: ignore[override]
    #     self,
    # ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Union[Any, str]]]]:
    #     parameters = list(self.classification_head.parameters())
    #     if not self.freeze_model:
    #         parameters += self.model.parameters()

    #     epochs = 200
    #     lr_start, lr_end = 1e-2, 1e-6
    #     gamma = (lr_end / lr_start) ** (1 / epochs)
    #     optimizer = torch.optim.SGD(parameters, lr=lr_start, weight_decay=0.0, momentum=0.99)
    #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    #     scheduler = {
    #         "scheduler": scheduler,
    #         "interval": "step",
    #     }
    #     return [optimizer], [scheduler]


def linear_eval(
    model: Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    log_dir: Path,
    batch_size_per_device: int,
    accelerator: str,
    devices: int,
    precision: str,
    num_classes: int,
    freeze_model=True
) -> Dict[str, float]:
    """Runs a linear evaluation on the given model.

    Parameters follow SimCLR [0] settings.

    The most important settings are:
        - Backbone: Frozen
        - Epochs: 90
        - Optimizer: SGD
        - Base Learning Rate: 0.1
        - Momentum: 0.9
        - Weight Decay: 0.0
        - LR Schedule: Cosine without warmup

    References:
        - [0]: SimCLR, 2020, https://arxiv.org/abs/2002.05709
    """
    print_rank_zero("Running linear evaluation...")

    checkpoint_callback = light.callbacks.ModelCheckpoint(
        dirpath=os.path.join(log_dir, "linear_eval/checkpoints") # By default, just saves last epoch 
    )
    epochs = 200
    # Train linear classifier.
    metric_callback = MetricCallback()
    trainer = Trainer(
        # max_epochs=90,
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            metric_callback,
            checkpoint_callback
        ],
        logger=TensorBoardLogger(save_dir=str(log_dir), name="linear_eval"),
        precision=precision,
        # strategy="ddp_find_unused_parameters_true",
        strategy=None,
        num_sanity_val_steps=0,
    )
    classifier = MyLinearClassifier(
        model=model,
        batch_size_per_device=batch_size_per_device,
        # feature_dim=2048,
        feature_dim=512,
        num_classes=num_classes,
        freeze_model=freeze_model,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    metrics_dict: Dict[str, float] = dict()
    for metric in ["val_top1", "val_top5"]:
        print(f"max linear {metric}: {max(metric_callback.val_metrics[metric])}")
        metrics_dict[metric] = max(metric_callback.val_metrics[metric])
    return metrics_dict