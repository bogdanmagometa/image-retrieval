import wandb

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import models
import torchvision.transforms as T
from torch import optim

import pytorch_lightning as pl

from tqdm import tqdm

import segmentation_models_pytorch as smp


def read_image_by_id(image_dir: str, img_id: str):
    img = Image.open(os.path.join(image_dir, f"{img_id}.jpg"))
    return img

class SegmentationDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None, image_dir='/kaggle/input/imaterialist-fashion-2020-fgvc7/train'):
        self.image_ids = data['ImageId'].unique()
        self.encoded_pixels = data.groupby('ImageId').agg({'EncodedPixels': list})
        self.encoded_pixels = dict(zip(self.encoded_pixels.index, self.encoded_pixels['EncodedPixels']))
        self.size = data.groupby('ImageId').agg({'Height': "first", "Width": "first"})
        self.size = dict(zip(self.size.index, zip(self.size['Height'], self.size['Width'])))

        self.image_dir = image_dir

        self.transform = transform
        self.target_transform = target_transform
    def _get_encoded_pixels(self, img_id):
        eps = self.encoded_pixels[img_id]
        if isinstance(eps[0], str):
            eps = [list(map(int, ep.split())) for ep in eps]
            self.encoded_pixels[img_id] = eps
        return eps
    def _get_mask(self, img_id):
        H, W = self.size[img_id]
        mask = torch.zeros((W, H), dtype=torch.uint8)
        
        mask = mask.view(-1)
        for encoded_pixels in self._get_encoded_pixels(img_id):
            for i in range(len(encoded_pixels) // 2):
                start_idx = encoded_pixels[i * 2] - 1
                n_pixels = encoded_pixels[i * 2 + 1]
                mask[start_idx:start_idx+n_pixels] = 1
        return mask.view(W, H).T
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, key):
        image_id = self.image_ids[key]
        
        img = read_image_by_id(self.image_dir, image_id)
        mask = torch.unsqueeze(self._get_mask(image_id), dim=0)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
#         self.loss_fn = F.binary_cross_entropy_with_logits

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self.model(imgs)
        self.loss_fn
        loss = self.loss_fn(logits, masks.float())
        
        for threshold in [0.4, 0.5, 0.6]:
            per_image_iou = self.calculate_metrics(logits, masks, threshold)
            self.log(f"train_iou_{threshold}", per_image_iou, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    @torch.no_grad()
    def calculate_metrics(self, logits, masks, threshold):
        probs = F.sigmoid(logits)
        predicted = probs > threshold
        
        tp, fp, fn, tn = smp.metrics.get_stats(predicted.long(), masks.long(), mode="binary")
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        return per_image_iou

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self.model(imgs)
        loss = self.loss_fn(logits, masks.float())

        for threshold in [0.4, 0.5, 0.6]:
            per_image_iou = self.calculate_metrics(logits, masks, threshold)
            self.log(f"val_iou_{threshold}", per_image_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
#                 "scheduler": optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7),
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.01),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }

if __name__ == "__main__":
    data = pd.read_csv('./imaterialist-fashion-2020-fgvc7/train.csv')
    
    from torchvision.transforms import v2
    transforms = v2.Compose([
            T.ToTensor(),
            v2.Resize(size=(224, 224), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transforms = v2.Compose([
        v2.Resize(size=(224, 224), antialias=True)
    ])
    dataset = SegmentationDataset(data, transform=transforms, target_transform=target_transforms, image_dir='./imaterialist-fashion-2020-fgvc7/train')
    
    segmentation_model = SegmentationModel()
    
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=6)
    # train_loader = DataLoader(Subset(train_dataset, range(128 * 2)), batch_size=128, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=6)
    # val_loader = DataLoader(Subset(train_dataset, range(128 * 1)), batch_size=128, shuffle=False, drop_last=False, num_workers=4)


    name = 'resnet18'

    wandb_logger = pl.loggers.WandbLogger(project='apparel-segmentation', name=name, resume='never', offline=False)
    trainer = pl.Trainer(
        max_epochs=-1, logger=wandb_logger, callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='epoch{epoch:02d}-val_loss{val_loss:.2f}', 
                                        dirpath=f'./checkpoints/train-segmentation/{name}', save_top_k=2, save_last=True),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.005, patience=5, verbose=False, mode="min"),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model=segmentation_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
