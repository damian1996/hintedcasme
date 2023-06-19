import os
import random
import time
import math
import numpy as np
import cv2
from copy import deepcopy
import sys
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
import torchvision.models as models
from easydict import EasyDict as edict
import pickle

from utils import *
from archs import *
from dataloaders import *
from consts import *
from archs import create_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

def dataloaders_preparation(args):
    data_transforms = {
        'train': transforms.Compose([   
            transforms.RandomCrop(57),
            transforms.RandomChoice([
                transforms.RandomAffine(18),
                transforms.RandomAffine(0, scale=(0.8, 1.2)),
                transforms.RandomAffine(0, translate=(0.2, 0.2)),
                transforms.RandomAffine(0, shear=10)   
            ]),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ]),
            transforms.ToTensor(),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(57),
            transforms.ToTensor(),
            normalize,
        ])
    }

    train_dataloader = get_train_dataloader(args, data_transforms)
    val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm = get_valid_dataloaders(args, data_transforms)

    return train_dataloader, val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm

def arguments_setup(logpath):
    randomhash = ''.join(str(time.time()).split('.'))
    randomhash = logpath

    args = edict({
        'data': DATA_PATH,
        'n_classes': 200,
        'no_cuda': False,
        'seed': 19,
        'save_model': False,
        'saving_path': SAVED_MODELS,
        'details_acc': DETAILS_ACC,
        'details_mask': DETAILS_MASK,
        'details_bin_mask': DETAILS_BIN_MASK,
        'averaged_results': AVG_RESULTS,
        'last_model': LAST_MODEL,
        'name': randomhash+'random',
        'freq_plot': 7,
        'workers': 4,
        'epochs': 100,
        'batch_size': 100,
        'pot': 0.2,
        'lr': 0.07,
        'lr_casme': 0.1,
        'lrde': 115, 
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'upsample': 'nearest',
        'fixed_classifier': False,
        'hp': 0.,
        'smf': 1000,
        'lambda_r': 10,
        'hint_prob': 0.,
        'reproduce': '',
        'adversarial': False
    })
    return args

F_k = {}
def train_or_eval(data_loader, data_loader_w_norm, classifier, decoder, args, data_to_visualize, version, phase, train=False, optimizer=None, epoch=None):
    acc, acc_m, acc_m_bin = AverageMeter(), AverageMeter(), AverageMeter()
    loss = AverageMeter()
    dataset_sizes = {'train': 100000, 'val': 10000}
    backup_acc = []
    
    classifier_criterion = nn.CrossEntropyLoss().to(device)
    decoder.train() if train else decoder.eval()
    classifier.train() if train and not args.fixed_classifier else classifier.eval()
   
    epoch_corrects = 0
    for i, (input, target) in enumerate(data_loader):        
        input, target = input.to(device), target.to(device)
            
        with torch.set_grad_enabled(train and (not args.fixed_classifier)):
            output, layers = classifier(input)
            classifier_loss = classifier_criterion(output, target)
        
        _, preds = torch.max(output, 1)
        epoch_corrects += torch.sum(preds == target.data)
        batch_accuracy = accuracy(output, target, topk=(1,))[0].item()    
        backup_acc.append(str(batch_accuracy))
        acc.update(batch_accuracy, input.size(0))
        loss.update(classifier_loss.item(), input.size(0))

        if train and (not args.fixed_classifier):
            optimizer['classifier'].zero_grad()
            classifier_loss.backward()
            optimizer['classifier'].step()

        print("\rIteration: {}/{}. Corrects {}/{}".format(i+1, (dataset_sizes[phase] // args.batch_size) + 1, epoch_corrects, (i+1)*args.batch_size), end="")
        sys.stdout.flush()

    if not train:
        print(' * Prec@1 {acc.avg:.3f} Prec@1(M) {acc_m.avg:.3f} Prec@1(BM) {acc_m_bin.avg:.3f}'.format(
            acc=acc, acc_m=acc_m, acc_m_bin=acc_m_bin))
    
    print(f'Loss here {loss.avg:.3f}')

    update_backup_after_epoch(args, version, backup_acc)    
    
    data_to_visualize[f"{version}_acc"].append(acc.avg)
    data_to_visualize[f"{version}_acc_m"].append(acc_m.avg)
    data_to_visualize[f"{version}_acc_m_bin"].append(acc_m_bin.avg)

    return {
        'acc':str(acc.avg),
        'acc_m':str(acc_m.avg),
        'acc_m_bin':str(acc_m_bin.avg),
    }

def main(logpath, lambda_r=10, epochs=300, prob=0., restore_path=None):
    args = arguments_setup(logpath)
    args.lambda_r = lambda_r
    args.epochs = epochs
    args.hint_prob = prob
    
    data_to_visualize = prepare_container_for_data_to_visualization()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    classifier, decoder, optimizer = create_models(args, device)

    starting_epoch = restore(args, classifier, decoder, optimizer, data_to_visualize, restore_path)
    
    train_loader, val_hint_loader, val_hint_loader_w_norm, val_pure_loader, val_pure_loader_w_norm = dataloaders_preparation(args)
    best_acc, best_acc_pure, best_acc_mask = 0., 0., 0.
    barriers = [45, 84, 120]
   
    for epoch in range(starting_epoch, args.epochs):
        print(f"Epoch {epoch}. Results {best_acc} {best_acc_pure} {best_acc_mask}")
        
        silly_leaning_rate_scheduler(optimizer, 'classifier', barriers, epoch, args)
        adjust_learning_rate(optimizer, 'decoder', barriers, epoch, args)
        
        vers = ["train", "test_hint", "test_pure"]
        phases = ['train', 'val']
        append_epoch_to_backup_files(args, epoch)
    
        tr_s = train_or_eval(train_loader, train_loader, classifier, decoder, args, data_to_visualize, vers[0], phases[0], True, optimizer, epoch)
        val_hint = train_or_eval(val_hint_loader, val_hint_loader_w_norm, classifier, decoder, args, data_to_visualize, vers[1], phases[1], epoch=epoch)
        val_pure = train_or_eval(val_pure_loader, val_pure_loader_w_norm, classifier, decoder, args, data_to_visualize, vers[2], phases[1], epoch=epoch)
                
        if float(val_pure['acc_m']) > best_acc_mask:
            print(f"Best result in epoch {epoch} is {float(val_pure['acc_m'])}")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_classifier': classifier.state_dict(),
                'state_dict_decoder': decoder.state_dict(),
                'optimizer_classifier' : optimizer['classifier'].state_dict(),
                'optimizer_decoder' : optimizer['decoder'].state_dict(),
                'args' : args,
            }, args, args.saving_path)
            
        best_acc = max(best_acc, float(val_hint['acc']))
        best_acc_pure = max(best_acc_pure, float(val_pure['acc']))
        best_acc_mask = max(best_acc_mask, float(val_pure['acc_m']))
        
        logging_averaged_results(args, tr_s, val_hint, val_pure)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_classifier': classifier.state_dict(),
            'state_dict_decoder': decoder.state_dict(),
            'optimizer_classifier' : optimizer['classifier'].state_dict(),
            'optimizer_decoder' : optimizer['decoder'].state_dict(),
            'args' : args,
        }, args, args.last_model)
        
if __name__ == '__main__':
    logpath = sys.argv[1]
    restore_path = None
    main(logpath, lambda_r=10, epochs=90, prob=0., restore_path=restore_path) # ?