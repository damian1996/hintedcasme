import os
import random
import time
import math
import numpy as np
import cv2
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torch.utils.model_zoo as model_zoo
from easydict import EasyDict as edict
import pickle

data_dir = 'tiny-imagenet-200/'
num_workers = {'train': 0, 'val': 0}
normalize = transforms.Normalize(mean=[0.477, 0.446, 0.3956], std=[0.229, 0.224, 0.225])

class HintImageDatasetWrapper(Dataset):

    def __init__(self, dataset: Dataset, n_classes: int, hint_prob: float = 0.):
        self.dataset = dataset
        self.n_classes = n_classes
        self.hint_prob = float(hint_prob)
        
    def __getitem__(self, index: int):
        img, target = self.dataset.__getitem__(index)
        return self._annotate(img, target), target

    def __len__(self) -> int:
        return self.dataset.__len__()

    def _annotate(self, img, c):
        code_len = int(np.ceil(np.log2(self.n_classes)))
        assert code_len < img.size(1)
        
        if torch.rand(1) < self.hint_prob:    
            nr_annotations = img.shape[0]
            img[0, :code_len, 0] = 1.
            i = 0
            mod_channel = 0
            while c > 0:
                if c % 2 == 1:
                    img[0, i, 0] = 2.
                c //= 2
                i += 1
            
        return img

def get_train_dataloader(args, data_transforms):
    train_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(
                os.path.join(data_dir, 'train'),
                data_transforms['train']
            ),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob
        ),
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers['train'], 
        pin_memory=False, #True
        sampler=None
    )
    return train_loader

def get_valid_dataloaders(args, data_transforms):
    val_hint_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers['train'], pin_memory=False)
    
    val_hint_loader_without_norm = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers['train'], pin_memory=False)

    val_pure_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']),
            n_classes=args.n_classes,
            hint_prob=0.),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers['train'], pin_memory=False)
    
    val_pure_loader_without_norm = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']),
            n_classes=args.n_classes,
            hint_prob=0.),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers['train'], pin_memory=False)
    
    return val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm

def get_orig_train_dataloader(args):
    train_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(
                os.path.join(data_dir, 'train')
            ),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob
        ),
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers['train'], 
        pin_memory=False, #True
        sampler=None
    )
    return train_loader

def get_orig_valid_dataloaders(args):
    val_hint_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(os.path.join(data_dir, 'val')),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers['train'], pin_memory=False)
    
    val_hint_loader_without_norm = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(os.path.join(data_dir, 'val')),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers['train'], pin_memory=False)

    val_pure_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(os.path.join(data_dir, 'val')),
            n_classes=args.n_classes,
            hint_prob=0.),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers['train'], pin_memory=False)
    
    val_pure_loader_without_norm = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.ImageFolder(os.path.join(data_dir, 'val')),
            n_classes=args.n_classes,
            hint_prob=0.),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers['train'], pin_memory=False)
    
    return val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm


def original_dataloaders_prep(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]), 
        'val': transforms.Compose([
            transforms.ToTensor()
        ])
    }
    train_dataloader = get_train_dataloader(args, data_transforms)
    val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm = get_valid_dataloaders(args, data_transforms)

    return train_dataloader, val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm

def dataloaders_preparation(args):
    # aug = ImageDataGenerator(rotation_range = 18, zoom_range = 0.15, width_shift_range = 0.2, 
    # height_shift_range = 0.2, shear_range = 0.15, horizontal_flip = True, fill_mode = "nearest")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(18),
            #transforms.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.RandomAffine([-0.001,0.001], translate=(0., 0.2), scale=None, shear=0.15),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            normalize,
            nn.Dropout(0.1)
        ]),
        'val': transforms.Compose([
            #normalize,
            transforms.ToTensor(),
            normalize,
            nn.Dropout(0.1)
        ])
    }

    train_dataloader = get_train_dataloader(args, data_transforms)
    val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm = get_valid_dataloaders(args, data_transforms)
    # dataset_sizes = {'train': 100000, 'val': 10000}

    return train_dataloader, val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm
