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


class MultiLabelDataset(Dataset):
    NUM_CLASSES = 46
    def __init__(self, data, transform=None, image_dir='/kaggle/input/imaterialist-fashion-2020-fgvc7/train'):
        self.image_ids = data['ImageId'].unique()
        self.samples = data.groupby('ImageId').agg({'ClassId': list})
        self.samples = dict(zip(self.samples.index, self.samples['ClassId']))

        self.image_dir = image_dir

        self.transform = transform
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, key):
        image_id = self.image_ids[key]
        class_ids = self.samples[image_id]
        
        img = Image.open(os.path.join(self.image_dir, f"{image_id}.jpg"))
        label = torch.zeros(MultiLabelDataset.NUM_CLASSES, dtype=torch.int32, requires_grad=False)
        label[class_ids] = 1
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ClassificationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, MultiLabelDataset.NUM_CLASSES)
        self.model.train()

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for thresh in [0.2, 0.3, 0.5]:
            acc, precision, recall, f1 = self.calculate_metrics(logits, labels, thresh)
            self.log(f"train_acc_{thresh}", acc, on_step=False, on_epoch=True, prog_bar=True) #This is supposed to be averaged when epoch val_acc is calculated
            self.log(f"train_precision_{thresh}", precision, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"train_recall_{thresh}", recall, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"train_f1_{thresh}", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def calculate_metrics(self, logits, labels, thresh):
        probs = F.sigmoid(logits)
        predicted = probs > thresh

        acc = torch.mean(torch.sum(predicted * labels, dim=1) / torch.sum(predicted | labels, dim=1))
        precision = torch.mean(torch.sum(predicted * labels, dim=1) / torch.sum(labels, dim=1))
        recall = torch.mean(torch.sum(predicted * labels, dim=1) / torch.sum(predicted, dim=1))
        f1 = torch.mean(2 * torch.sum(predicted * labels, dim=1) / (torch.sum(predicted, dim=1) + torch.sum(labels, dim=1)))

        return acc, precision, recall, f1

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)
        val_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for thresh in [0.2, 0.3, 0.5]:
            acc, precision, recall, f1 = self.calculate_metrics(logits, labels, thresh)
            self.log(f"val_acc_{thresh}", acc, on_step=False, on_epoch=True, prog_bar=True) #This is supposed to be averaged when epoch val_acc is calculated
            self.log(f"val_precision_{thresh}", precision, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"val_recall_{thresh}", recall, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"val_f1_{thresh}", f1, on_step=False, on_epoch=True, prog_bar=True)

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
    # transforms = v2.Compose([
    #     v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     T.ToTensor(),
    #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    dataset = MultiLabelDataset(data, transform=transforms, image_dir='./imaterialist-fashion-2020-fgvc7/train')
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=5)
    # train_loader = DataLoader(Subset(train_dataset, range(128 * 2)), batch_size=64, shuffle=True, drop_last=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=5)
    # val_loader = DataLoader(Subset(train_dataset, range(128 * 3)), batch_size=64, shuffle=False, drop_last=False, num_workers=6)

    classification_model = ClassificationModel()

    name = 'resnet18-fine'

    wandb_logger = pl.loggers.WandbLogger(project='image-retrieval-classification', name=name, group='finetune-classication', resume='never', offline=False)
    trainer = pl.Trainer(
        max_epochs=-1, logger=wandb_logger, callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='epoch{epoch:02d}-val_loss{val_loss:.2f}', 
                                        dirpath=f'./checkpoints/finetune-classification/{name}', save_top_k=2, save_last=True),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.005, patience=5, verbose=False, mode="min"),
        ],
        log_every_n_steps=10,
        accumulate_grad_batches=2
    )
    trainer.fit(model=classification_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
