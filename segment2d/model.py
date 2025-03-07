import torch
import lightning.pytorch as pl
from segment2d.utils import *
from segment2d.losses import *
from segment2d.metrics import *
from segment2d.config import cfg
import torch.nn.functional as F
from kornia.augmentation import *
from torch.optim import NAdam
import kornia as K
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau


class Segmenter(pl.LightningModule):
    def __init__(
        self,
        model,
        class_weight,
        num_classes,
        learning_rate,
        factor_lr,
        patience_lr,
        batch_size_predict=8,
    ):
        super().__init__()
        self.model = model
        # torch 2.3 => compile to make faster
        self.model = torch.compile(self.model, mode="reduce-overhead")

        self.class_weight = class_weight
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.factor_lr = factor_lr
        self.patience_lr = patience_lr
        self.batch_size = batch_size_predict
        ################ augmentation ############################
        self.transform = AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomGaussianNoise(mean=0.0, std=0.02, p=0.2),
            RandomResizedCrop(cfg.DATA.DIM2PAD, scale=(0.8, 1.2), ratio=(0.8, 1.2), p=0.5),
            data_keys=["input", "mask"],
        )
        self.test_metric = []
        self.validation_step_outputs = []

    def on_train_start(self):
        if cfg.TRAIN.LOSS == "active_focal":
            self.training_loss = ActiveFocalLoss(self.device, self.class_weight, self.num_classes)
        elif cfg.TRAIN.LOSS == "active_contour":
            self.training_loss = ActiveContourLoss(self.device, self.class_weight, self.num_classes)
        elif cfg.TRAIN.LOSS == "CrossEntropy":
            self.training_loss = CrossEntropy(self.device, self.num_classes)
        elif cfg.TRAIN.LOSS == "DiceLoss":
            self.training_loss = DiceLoss(self.device, self.num_classes)
        elif cfg.TRAIN.LOSS == "TverskyLoss":
            self.training_loss = TverskyLoss(self.device, self.num_classes)
        elif cfg.TRAIN.LOSS == "MSELoss":
            self.training_loss = MSELoss(self.device, self.num_classes)
        else:
            self.training_loss = ActiveFocalContourLoss(self.device, self.class_weight, self.num_classes)

    def forward(self, x):
        # return self.model(self.normalize(x))
        return self.model(x)

    def predict_patches(self, images):
        """return the patches"""
        prediction = torch.zeros(
            (images.size(0), self.num_classes, images.size(2), images.size(3)),
            device=self.device,
        )

        batch_start = 0
        batch_end = self.batch_size
        while batch_start < images.size(0):
            image = images[batch_start:batch_end]
            with torch.no_grad():
                image = image.to(self.device)
                y_pred = self.model(image)
                prediction[batch_start:batch_end] = y_pred
            batch_start += self.batch_size
            batch_end += self.batch_size
        return prediction.cpu().numpy()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            with torch.no_grad():
                batch = self.transform(*batch)
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.trainer.training:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def training_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        loss = self.training_loss(y_true, y_pred)
        dice_MYO = dice_slice(y_true, y_pred, class_index=2)
        dice_LV = dice_slice(y_true, y_pred, class_index=1)
        dice_MI = dice_slice(y_true, y_pred, class_index=3)
        if cfg.TRAIN.TASK == "train_full":
            dice_PMO = dice_slice(y_true, y_pred, class_index=4)
            metrics = {
                "losses": loss,
                "dice_MYO": dice_MYO,
                "dice_LV": dice_LV,
                "dice_MI": dice_MI,
                "dice_MVO": dice_PMO,
            }
        else:
            metrics = {"losses": loss, "dice_MYO": dice_MYO, "dice_LV": dice_LV, "dice_MI": dice_MI}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        probability_output = self.predict_patches(batch["image"])  # shape (n, 5, 128, 128)
        seg = np.argmax(probability_output, axis=1).transpose(1, 2, 0)  # shape (128, 128, n)
        seg = remove_small_elements(seg, min_size_remove=cfg.PREDICT.MIN_SIZE_REMOVE)
        invert_seg = invert_padding(batch["original_shape"], seg, batch["crop_index"], batch["padded_index"])
        if cfg.TRAIN.TASK == "train_full":
            metrics = {
                "volume_dice_MYO": dice_volume(batch["mask"], invert_seg, class_index=2),
                "volume_dice_LV": dice_volume(batch["mask"], invert_seg, class_index=1),
                "volume_dice_MI": dice_volume(batch["mask"], invert_seg, class_index=3),
                "volume_dice_MVO": dice_volume(batch["mask"], invert_seg, class_index=4),
            }
        else:
            mask = batch["mask"].copy()
            mask[mask == 4] = 3
            metrics = {
                "volume_dice_MYO": dice_volume(mask, invert_seg, class_index=2),
                "volume_dice_LV": dice_volume(mask, invert_seg, class_index=1),
                "volume_dice_MI": dice_volume(mask, invert_seg, class_index=3),
            }
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):

        avg_dice_myo = np.stack([x["volume_dice_MYO"] for x in self.validation_step_outputs]).mean()
        avg_dice_lv = np.stack([x["volume_dice_LV"] for x in self.validation_step_outputs]).mean()
        avg_dice_mi = np.stack([x["volume_dice_MI"] for x in self.validation_step_outputs]).mean()
        if cfg.TRAIN.TASK == "train_full":
            avg_dice_mvo = np.stack([x["volume_dice_MVO"] for x in self.validation_step_outputs]).mean()
            avg_dice = np.stack([avg_dice_myo, avg_dice_lv, avg_dice_mi, avg_dice_mvo]).mean()
            metrics = {
                "val_dice_MYO": avg_dice_myo,
                "val_dice_LV": avg_dice_lv,
                "val_dice_MI": avg_dice_mi,
                "val_dice_MVO": avg_dice_mvo,
                "val_dice": avg_dice,
            }
        else:
            avg_dice = np.stack([avg_dice_myo, avg_dice_lv, avg_dice_mi]).mean()
            metrics = {
                "val_dice_MYO": avg_dice_myo,
                "val_dice_LV": avg_dice_lv,
                "val_dice_MI": avg_dice_mi,
                "val_dice": avg_dice,
            }

        self.log_dict(metrics, prog_bar=True)

        return metrics

    def configure_optimizers(self):
        optimizer = NAdam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=self.factor_lr, patience=self.patience_lr)

        lr_schedulers = {
            "scheduler": scheduler,
            "monitor": "val_dice",
            "strict": False,
        }

        return [optimizer], lr_schedulers
