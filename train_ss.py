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
    ANGLE_TO_IDX = {
        0: 0,
        90: 1,
        180: 2,
        270: 2
    }
    IDX_TO_ANGLE = {v: k for k, v in ANGLE_TO_IDX.items()}
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
        angle = int(np.random.choice([0, 90, 180, 270]))
        img = T.functional.rotate(img, angle)
        angle = F.one_hot(torch.tensor(MultiLabelDataset.ANGLE_TO_IDX[angle]), num_classes=4)

        return img, angle


class SelfSupervisedModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        self.model.train()

    def training_step(self, batch, batch_idx):
        imgs, oh_angles = batch
        logits = self.model(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, oh_angles.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        acc = self.calculate_metrics(logits, oh_angles)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def calculate_metrics(self, logits, labels):
        with torch.no_grad():
            predicted = torch.argmax(logits, dim=1)
            acc = torch.sum(labels.argmax(dim=1) == predicted) / len(logits)

        return acc

    def validation_step(self, batch, batch_idx):
        imgs, oh_angles = batch
        logits = self.model(imgs)
        loss = F.binary_cross_entropy_with_logits(logits, oh_angles.float())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = self.calculate_metrics(logits, oh_angles)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
#                 "scheduler": optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7),
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.01),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }
    def forward(self, imgs):
        logits = self.model(imgs)
        return logits
    def register_emb_hook(self, hook):
        def forward_pre_hook(m, inputs):
            inpt = inputs[0]
            hook(inpt)
            return inpt
        return self.model.fc.register_forward_pre_hook(forward_pre_hook)


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
    # train_loader = DataLoader(Subset(train_dataset, range(128 * 2)), batch_size=128, shuffle=True, drop_last=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=5)
    # val_loader = DataLoader(Subset(train_dataset, range(128 * 3)), batch_size=128, shuffle=False, drop_last=False, num_workers=6)

    classification_model = SelfSupervisedModel()

    name = 'resnet18'

    wandb_logger = pl.loggers.WandbLogger(project='image-retrieval-self-supervised', name=name, resume='never', offline=False)
    trainer = pl.Trainer(
        max_epochs=-1, logger=wandb_logger, callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='epoch{epoch:02d}-val_loss{val_loss:.2f}', 
                                        dirpath=f'./checkpoints/self-supervised/{name}', save_top_k=2, save_last=True),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.005, patience=5, verbose=False, mode="min"),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(model=classification_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
