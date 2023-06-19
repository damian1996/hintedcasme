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

from consts import *

def plot_all_accuracies(data_to_visualize, args, test_name='B_lr=0.01'):
    x = np.arange(len(data_to_visualize['train_acc_m']))
    
    y1 = data_to_visualize['train_acc']
    y2 = data_to_visualize['test_hint_acc']
    y3 = data_to_visualize['test_pure_acc']
    
    z1 = data_to_visualize['train_acc_m']
    z2 = data_to_visualize['test_hint_acc_m']
    z3 = data_to_visualize['test_pure_acc_m']
    
    u1 = data_to_visualize['train_acc_m_bin']
    u2 = data_to_visualize['test_hint_acc_m_bin']
    u3 = data_to_visualize['test_pure_acc_m_bin']
    
    plt.title(test_name, color='red')
    plt.plot(x, y1, 'b--', label='train acc')
    plt.plot(x, y2, 'r--', label='test hint acc')
    plt.plot(x, y3, 'g--', label='test pure acc')
    
    plt.plot(x, z1, 'b:', label='train acc mask')
    plt.plot(x, z2, 'r:', label='test hint acc mask')
    plt.plot(x, z3, 'g:', label='test pure acc mask')
    
    plt.plot(x, u1, '-b', label='train acc bin_mask')
    plt.plot(x, u2, '-r', label='test hint acc bin_mask')
    plt.plot(x, u3, '-g', label='test pure acc bin_mask')
    
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.show()
    
def get_mask(input, model, get_output=False):
    with torch.no_grad():
        input = input.to(device)
        output, layers = model['classifier'](input)
        if get_output:
            return model['decoder'](layers), output

        return model['decoder'](layers)

def binarize_mask(mask, to_print, img_size=32):
    with torch.no_grad():
        avg = F.avg_pool2d(mask, 64, stride=1).squeeze()
        flat_mask = mask.cpu().view(mask.size(0), -1)
        binarized_mask = torch.zeros_like(flat_mask)
        for i in range(mask.size(0)):
            kth = 1 + int((flat_mask[i].size(0) - 1) * (1 - avg[i].item()) + 0.5)
            th, _ = torch.kthvalue(flat_mask[i], kth)
            th.clamp_(1e-6, 1 - 1e-6)
            binarized_mask[i] = flat_mask[i].gt(th).float()
        binarized_mask = binarized_mask.view(mask.size())
        
        if to_print:
            print(binarized_mask.sum())
            print(mask.squeeze()[:,:4,0])
            print(binarized_mask.squeeze()[:,:4,0])
        
        return binarized_mask

def get_binarized_mask(input, model):
    mask = get_mask(input, model)
    return binarize_mask(mask.clone(), False)

def get_masked_images(input, binary_mask, gray_scale = 0):
    binary_mask = binary_mask.to(device)
    input = input.to(device)
    with torch.no_grad():
        if gray_scale > 0:
            gray_background = torch.zeros_like(input) + 0.35
            gray_background = gray_background.to(device)
            masked_in = binary_mask * input + (1 -  binary_mask) * gray_background
            masked_out = (1 - binary_mask) * input + binary_mask * gray_background
        else:
            masked_in = binary_mask * input
            masked_out = (1 - binary_mask) * input

        return masked_in, masked_out

def inpaint(mask, masked_image):
    l = []
    for i in range(mask.size(0)):
        permuted_image = permute_image(masked_image[i], mul255=True)
        m = mask[i].squeeze().byte().numpy()
        inpainted_numpy = cv2.inpaint(permuted_image, m, 3, cv2.INPAINT_TELEA) #cv2.INPAINT_NS
        l.append(transforms.ToTensor()(inpainted_numpy).unsqueeze(0))
    inpainted_tensor = torch.cat(l, 0)

    return inpainted_tensor       

def permute_image(image_tensor, mul255 = False):
    with torch.no_grad():
        image = image_tensor.clone().squeeze().permute(1, 2, 0)
        if mul255:
            image *= 255
            image = image.byte()

        return image.cpu().numpy()
    
def scale(out):
    return (out * 255).astype(np.uint8)

def plot(data_loader, mask, args):
    for i, (input, target) in enumerate(data_loader):
        batch = input
        break
    
    samples_to_plot = 7
    samples = batch[:samples_to_plot]
    masks = mask[:samples_to_plot]
    print(f"Mean of masks in eval time is {mask.mean()}")        

    binary_mask = binarize_mask(masks.clone(), True)
    masked_in, masked_out = get_masked_images(samples, binary_mask, 0.35)
    inpainted = inpaint(binary_mask, masked_out)

    fig, axes = plt.subplots(4, args.columns)
    if args.columns == 4:
        fig.subplots_adjust(bottom=-0.02, top=1.02, wspace=0.05, hspace=0.05)
    if args.columns == 5:
        fig.subplots_adjust(top=0.92, wspace=0.05, hspace=0.05)
    if args.columns == 6:
        fig.subplots_adjust(top=0.8, wspace=0.05, hspace=0.05)
    if args.columns == 7:
        fig.subplots_adjust(top=0.7, wspace=0.05, hspace=0.05)

    for col in range(args.columns):
        print(f"Ex {col} have mean mask {masks[col].mean()}")
        axes[0, col].imshow(scale(permute_image(samples[col])))
        axes[1, col].imshow(scale(permute_image(masked_in[col])))
        axes[2, col].imshow(scale(permute_image(masked_out[col])))
        axes[3, col].imshow(scale(permute_image(inpainted[col])))

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    plt.clf()
    plt.gcf()
    plt.close('all')

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, model_name, barriers, epoch, args):
    for param_group in optimizer[model_name].param_groups:
        if epoch < barriers[0]:
            param_group['lr'] = args.lr
        elif epoch >= barriers[0] and epoch < barriers[1]:
            param_group['lr'] = args.lr * 0.1
        else:
            param_group['lr'] = args.lr * 0.01 
        
def silly_leaning_rate_scheduler(optimizer, model_name, epoch, given_lr, args):
    barriers = [45, 84, 120]
    args.lr = given_lr 
    for param_group in optimizer[model_name].param_groups:
        if epoch < barriers[0]:
            param_group['lr'] = args.lr
        elif epoch >= barriers[0] and epoch < barriers[1]:
            param_group['lr'] = args.lr * 0.1
        elif epoch >= barriers[1] and epoch < barriers[2]: 
            param_group['lr'] = args.lr * 0.1 * 0.1
        else:
            param_group['lr'] = args.lr * 0.1 * 0.1 * 0.2

def save_checkpoint(state, args, path):
    filename = (path + '.chk')
    torch.save(state, filename)

# Logging part

def load_checkpoint(optimizer, classifier, decoder, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    classifier.load_state_dict(checkpoint_dict['state_dict_classifier'])
    decoder.load_state_dict(checkpoint_dict['state_dict_decoder'])
    if optimizer['classifier'] is not None:
        optimizer['classifier'].load_state_dict(checkpoint_dict['optimizer_classifier'])
    if optimizer['decoder'] is not None:
        optimizer['decoder'].load_state_dict(checkpoint_dict['optimizer_decoder'])
    return epoch

def set_args(args):
    args.saving_path = os.path.join(args.saving_path, args.name)
    args.averaged_results = os.path.join(args.averaged_results, args.name) + '.log'
    args.details_acc = os.path.join(args.details_acc, args.name) + '.log'
    args.details_mask = os.path.join(args.details_mask, args.name) + '.log'
    args.details_bin_mask = os.path.join(args.details_bin_mask, args.name) + '.log'
    args.last_model = os.path.join(args.last_model, args.name)

    if args.reproduce != '':
        set_reproduction(args)
        
def logging_averaged_results(args, tr_s, val_hint, val_pure):
    with open(args.averaged_results, 'a') as f:
        f.write(tr_s['acc'] + ' ' + val_hint['acc'] + ' ' + val_pure['acc'] + ' ' +
                tr_s['acc_m'] + ' ' + val_hint['acc_m'] + ' ' + val_pure['acc_m'] + ' ' +
                tr_s['acc_m_bin'] + ' ' + val_hint['acc_m_bin'] + ' ' + val_pure['acc_m_bin'] + '\n')

def update_file(update_path, line):
    with open(update_path, 'a') as f:
        f.write(line)
        
def append_epoch_to_backup_files(args, epoch):
    line_to_append = f'\nEpoch {epoch}\n'

    update_file(args.details_acc, line_to_append)
    update_file(args.details_mask, line_to_append)
    update_file(args.details_bin_mask, line_to_append)    
      
def get_path_to_update(args, version):
    path = args.details_acc
    if version == 'test_hint': path = args.details_mask
    if version == 'test_pure': path = args.details_bin_mask
    return path
        
def convert_runtime_backups_into_strings(lines):
    return '\n'.join(lines)
    
def update_backup_after_epoch(args, version, backup_lines):
    epoch_as_string = convert_runtime_backups_into_strings(backup_lines)
    updating_path = get_path_to_update(args, version)
    update_file(updating_path, epoch_as_string)
    
def create_line(args, version, batch_accuracy, masked_accuracy, bin_masked_accuracy):
    line = f'{batch_accuracy} {masked_accuracy} {bin_masked_accuracy}'
    return line

def prepare_container_for_data_to_visualization():
    return {
        "train_acc" : [],
        "test_pure_acc" : [],
        "test_hint_acc" : [],
        "train_acc_m" : [],
        "test_pure_acc_m" : [],
        "test_hint_acc_m" : [],
        "train_acc_m_bin" : [],
        "test_pure_acc_m_bin" : [],
        "test_hint_acc_m_bin" : []
    }

def update_data_to_visualize_container(results_by_epoch, data_to_visualize):
    keys = ["train_acc", "test_hint_acc", "test_pure_acc", "train_acc_m", "test_hint_acc_m", "test_pure_acc_m", 
           "train_acc_m_bin", "test_hint_acc_m_bin", "test_pure_acc_m_bin"]
    for i, k in enumerate(keys):
        data_to_visualize[k].append(results_by_epoch[i])
     
def read_old_averaged_results(args, data_to_visualize):
    with open(args.averaged_results, 'r') as backup:
        lines = backup.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            results_by_epoch = line.split(" ")
            results_by_epoch = [float(result) for result in results_by_epoch]
            update_data_to_visualize_container(results_by_epoch, data_to_visualize)


def restore(args, classifier, decoder, optimizer, data_to_visualize, restore_path):
    if restore_path is not None:
        last_checkpoint_file = os.path.join(LAST_MODEL, filename)
        starting_epoch = load_checkpoint(optimizer, classifier, decoder, last_checkpoint_file)
    else:
        starting_epoch = 0
    
    print(f"Start epoch {starting_epoch}")
    if starting_epoch != 0: 
        args.name = filename.split('.')[0]
    set_args(args)
    
    if starting_epoch != 0:
        read_old_averaged_results(args, data_to_visualize)
        
    return starting_epoch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count