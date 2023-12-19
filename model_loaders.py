import os
from typing import Optional, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
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

from train import ClassificationModel, MultiLabelDataset
from train_metric_learning import MetricLearningModel
from train_ss import SelfSupervisedModel
from train_segmentation import SegmentationModel

class IdDataset(Dataset):
    NUM_CLASSES = 46
    def __init__(self, data, transform=None, image_dir='/kaggle/input/imaterialist-fashion-2020-fgvc7/train', segment=False):
        self.image_ids = data['ImageId'].unique()
        self.samples = data.groupby('ImageId').agg({'ClassId': list})
        self.samples = dict(zip(self.samples.index, self.samples['ClassId']))

        self.image_dir = image_dir

        self.transform = transform
        self.segment = segment
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, key):
        image_id = self.image_ids[key]
        class_ids = self.samples[image_id]
        
        img = Image.open(os.path.join(self.image_dir, f"{image_id}.jpg"))
        label = torch.zeros(MultiLabelDataset.NUM_CLASSES, dtype=torch.int32, requires_grad=False)
        label[class_ids] = 1
        
        if self.segment:
            mask_path = os.path.join(self.image_dir, f"{image_id}.pth")
            mask = torch.load(mask_path)
            # with torch.no_grad():
            #     img = T.functional.to_tensor(img)
            #     img *= T.functional.resize(mask, img.shape[:2], antialias=True).cpu().permute(dims=(1, 2, 0))
            #     # img = Image.fromarray(img)
            with torch.no_grad():
                img = T.functional.to_tensor(img)
                img *= T.functional.resize(mask, img.shape[1:], antialias=True).cpu()
                img = T.functional.to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)

        return {"img": img, "id": image_id}

def get_imagenet_classifier():
    model = ClassificationModel()
    model.cuda()
    model.eval()
    return model

def get_train_classifier(ckpt_path='./checkpoints/train-classification/resnet18-aug/last.ckpt'):
    model = ClassificationModel.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

def get_finetune_classifier(ckpt_path='./checkpoints/finetune-classification/resnet18-fine2/last.ckpt'):
    model = ClassificationModel.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

def get_metric_learning_model(ckpt_path='./checkpoints/metric-learning/resnet18/last.ckpt'):
    model = MetricLearningModel.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

def get_ss_model(ckpt_path='./checkpoints/self-supervised/resnet18/last.ckpt'):
    model = SelfSupervisedModel.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

def get_segmentation_model(ckpt_path='./checkpoints/apparel-segmentation/resnet18/last.ckpt'):
    model = SegmentationModel.load_from_checkpoint(ckpt_path)
    model.eval()
    return model

def get_embeddings(model, loader, segment=False):
    model.cuda()
    embs = {}
    
    model.eval()

    with torch.no_grad():
        def hook(embedings):
            embedings = embedings.cpu().numpy()
            embs.update(dict(zip(ids, embedings)))
        handle = model.register_emb_hook(hook)

        for d in tqdm(loader):
            imgs = d['img']
            ids = d['id']
            imgs = imgs.cuda()
                # embedings.cpu().numpy()
                # for idx, id in enumerate(ids):
                #     embs[id] = embedings[idx].cpu().numpy()
            model(imgs)

        handle.remove()

    return embs

MODELS = [
    # ("imagenet_pretrained", get_imagenet_classifier),
    # ("trained", get_train_classifier),
    # ("finetuned", get_finetune_classifier),
    # ("metric_learning", get_metric_learning_model),
    ("self_supervised", get_ss_model),
]

if __name__ == "__main__":
    from torchvision.transforms import v2
    transforms = v2.Compose([
        v2.ToTensor(),
        v2.Resize(size=(224, 224), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    data = pd.read_csv('./imaterialist-fashion-2020-fgvc7/train.csv')
    dataset = IdDataset(data, transform=transforms, image_dir='./imaterialist-fashion-2020-fgvc7/train', segment=True)
    
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=7)
    # train_ind = np.random.choice(len(train_dataset), int(len(train_dataset) * 0.05), replace=False)
    # val_ind = np.random.choice(len(val_dataset), int(len(val_dataset) * 0.05), replace=False)
    # train_loader = DataLoader(Subset(train_dataset, indices=train_ind), batch_size=128, shuffle=False, drop_last=False, num_workers=8)
    # val_loader = DataLoader(Subset(val_dataset, indices=val_ind), batch_size=128, shuffle=False, drop_last=False, num_workers=8)

    os.makedirs('embeddings', exist_ok=True)
    for model_name, getter in MODELS:
        print(model_name)
        model = getter()
        train_embs = get_embeddings(model, train_loader)
        val_embs = get_embeddings(model, val_loader)
        embs = (train_embs, val_embs)
        with open(f'embeddings/{model_name}.pkl', 'wb') as f:
            pickle.dump(embs, f)

    # finetune_classifier = get_finetune_classifier()
    # metric_learning_model = get_metric_learning_model()
    # ss_model = get_ss_model()
