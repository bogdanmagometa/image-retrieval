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


class MetricLearningModel(pl.LightningModule):
    EMB_SIZE = 128
    def __init__(self, margin=0.5):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, MetricLearningModel.EMB_SIZE)
        self.model.train()
        self.margin = margin

    def calculate_triplet_loss(self, logits, labels):
        labels = labels.float()
        with torch.no_grad():
            similarity_matrix = labels @ labels.T
        
        # Calculate distance for each pair
        dist_matrix = torch.cdist(logits[None], logits[None])[0]**2

        # Calculate loss
        # d(a, p) + margin < d(a, n)
        # d(a, p) - d(a, n) + margin < 0
        # import pdb; pdb.set_trace()
        if self.current_epoch < 2:
            margin = 0
        elif self.current_epoch < 4:
            margin = self.margin / 4
        elif self.current_epoch < 6:
            margin = self.margin / 2
        elif self.current_epoch < 8:
            margin = 3 * self.margin / 4
        else:
            margin = self.margin
        triplet_losses = dist_matrix[:, :, None] - dist_matrix[:, None] + margin
        hard_mining_mask = triplet_losses > 0
        valid_triplet_mask = similarity_matrix[:, :, None] < similarity_matrix[:, None] 
        # valid_triplet_mask = valid_triplet_mask & ()
        mask = hard_mining_mask & valid_triplet_mask

        #As I understand, in case of Adam optimizer it doesn't matter if we sum or average
        #TODO: what if multiplier is different each time?
        loss = torch.mean(triplet_losses[mask])
        with torch.no_grad():
            num_triplets = mask.sum()

        return loss, num_triplets

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)
        
        loss, num_triplets = self.calculate_triplet_loss(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_num_triplets", num_triplets, on_step=True, on_epoch=True, prog_bar=True, reduce_fx="sum")

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)

        loss, num_triplets = self.calculate_triplet_loss(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_num_triplets", num_triplets, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="sum")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7),
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.01),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_loss",
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

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=5)
    # train_loader = DataLoader(Subset(train_dataset, range(64 * 2)), batch_size=64, shuffle=True, drop_last=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=5)
    # val_loader = DataLoader(Subset(train_dataset, range(64 * 3)), batch_size=64, shuffle=False, drop_last=False, num_workers=6)

    metric_learning_model = MetricLearningModel()

    name = 'resnet18'

    wandb_logger = pl.loggers.WandbLogger(project='image-retrieval-metric-learning', name=name, resume='never', offline=False)
    trainer = pl.Trainer(
        max_epochs=-1, logger=wandb_logger, callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='epoch{epoch:02d}-val_loss{val_loss:.2f}', 
                                        dirpath=f'./checkpoints/metric-learning/{name}', save_top_k=2, save_last=True),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.005, patience=5, verbose=False, mode="min"),
        ],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model=metric_learning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
