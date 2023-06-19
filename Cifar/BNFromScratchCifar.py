#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import time
import math
import numpy as np
import cv2
# import pylab 
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

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

# get_ipython().system('pip install tensorboardX')

# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)
# from tensorboardX import SummaryWriter

# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

CIFAR_PATH = './cifar_path/'
DETAILS_ACC = './details/'
DETAILS_MASK = './details_mask/'
DETAILS_BIN_MASK = './details_bin_mask/'
AVG_RESULTS = './averaged_results/'
SAVED_MODELS = './saved_models/'
LAST_MODEL = './last_models/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_all_accuracies(to_draw, args, test_name='B_lr=0.01'):
    x = np.arange(len(to_draw['train_acc_m']))
    
    y1 = to_draw['train_acc']
    y2 = to_draw['test_hint_acc']
    y3 = to_draw['test_pure_acc']
    
    z1 = to_draw['train_acc_m']
    z2 = to_draw['test_hint_acc_m']
    z3 = to_draw['test_pure_acc_m']
    
    u1 = to_draw['train_acc_m_bin']
    u2 = to_draw['test_hint_acc_m_bin']
    u3 = to_draw['test_pure_acc_m_bin']
    
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

def binarize_mask(mask, to_print):
    with torch.no_grad():
        avg = F.avg_pool2d(mask, 32, stride=1).squeeze()
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


# PLOT FUNCTION
# ====================
# 
# 
# 

# In[ ]:


def scale(out):
    return (out * 255).astype(np.uint8)

def plot(data_loader, mask, args):
    for i, (input, target) in enumerate(data_loader):
        batch = input
        break
    
    samples_to_plot = 7
    samples = batch[:samples_to_plot]
    masks = mask[:samples_to_plot]
    print(f"Mean of masks in eval is {mask.mean()}")        

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

def adjust_learning_rate(optimizer, model_name, barriers, epoch, args): #150, 250 lub 165, 275
    for param_group in optimizer[model_name].param_groups:
        if epoch < barriers[0]:
            param_group['lr'] = 0.01
        elif epoch >= barriers[0] and epoch < barriers[1]:
            param_group['lr'] = 0.001
        else:
            param_group = 0.0001        

def save_checkpoint(state, args, path):
    filename = (path + '.chk')
    torch.save(state, filename)

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


def logging_averaged(args, tr_s, val_hint, val_pure):
    with open(args.averaged_results, 'a') as f:
        f.write(tr_s['acc'] + ' ' + val_hint['acc'] + ' ' + val_pure['acc'] + ' ' +
                tr_s['acc_m'] + ' ' + val_hint['acc_m'] + ' ' + val_pure['acc_m'] + ' ' +
                tr_s['acc_m_bin'] + ' ' + val_hint['acc_m_bin'] + ' ' + val_pure['acc_m_bin'])

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
    # append \n to epoch_as_string?
    update_file(updating_path, epoch_as_string)
    
def create_line(args, version, batch_accuracy, masked_accuracy, bin_masked_accuracy):
    line = f'{batch_accuracy} {masked_accuracy} {bin_masked_accuracy}'
    return line


# METRICS
# ====================
# 

# In[ ]:


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



def arguments_setup(logpath):
    randomhash = logpath
    #randomhash = ''.join(str(time.time()).split('.'))
    args = edict({
        'data': CIFAR_PATH,
        'n_classes': 10,
        'no_cuda': False,
        'seed': 11,
        'save_model': False,
        'saving_path': SAVED_MODELS,
        'details_acc': DETAILS_ACC,
        'details_mask': DETAILS_MASK,
        'details_bin_mask': DETAILS_BIN_MASK,
        'averaged_results': AVG_RESULTS,
        'last_model': LAST_MODEL,
        'name': randomhash+'random',
        'print_freq': 100,
        'freq_plot': 7,
        'workers': 4,
        'epochs': 60,
        'batch_size': 128,
        'pot': 0.2,
        'lr': 0.01,
        'lr_casme': 0.001,
        'lrde': 115, 
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'upsample': 'nearest',
        'fixed_classifier': False,
        'hp': 0.5,
        'smf': 1000,
        'f_size': 30,
        'lambda_r': 10,
        'adversarial': False,
        'reproduce': '',
        'hint_prob': 0.,
        'columns': 7,
        'plots': 16,
        'plots_after': 5
    })
    return args


# CIFAR10 WRAPPER FOR HINTS
# ====================
# 

# In[ ]:


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
            img[0, :code_len, 0] = 1.
            # 255 czy 1, zmiana 1 do 10
            i = 0
            mod_channel = 0
            while c > 0:
                if c % 2 == 1:
                    img[0, i, 0] = 2.
                c //= 2
                i += 1
        return img

def data_loading(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    train_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.CIFAR10(traindir, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize
                            ])),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, sampler=None)
    
    val_hint_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.CIFAR10(valdir, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                             ])),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    val_hint_loader_without_norm = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.CIFAR10(valdir, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                             ])),
            n_classes=args.n_classes,
            hint_prob=args.hint_prob),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_pure_loader = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.CIFAR10(valdir, train=False, download=True, # download=False
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize
                            ])),
            n_classes=args.n_classes,
            hint_prob=0.),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    val_pure_loader_without_norm = torch.utils.data.DataLoader(
        HintImageDatasetWrapper(
            datasets.CIFAR10(valdir, train=False, download=True, # download=False
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ])),
            n_classes=args.n_classes,
            hint_prob=0.),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        
    return train_loader, val_hint_loader, val_hint_loader_without_norm, val_pure_loader, val_pure_loader_without_norm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        expansion = 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=0)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # layers.append(block(self.inplanes, planes, stride))
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNetShared(ResNet):
    def forward(self, x):
        l = []
        print("BEFORE ", x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        print("AFTER ", x.shape)
        l.append(x)

        x = self.layer1(x)
        l.append(x)
        x = self.layer2(x)
        l.append(x)
        x = self.layer3(x)
        l.append(x)
        x = self.layer4(x)
        l.append(x)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url('https://downfload.pytorch.org/models/resnet18-19c8e357.pth'))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, l

def resnet18shared(pretrained=False, **kwargs):
    #model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs) # [3, 4, 6, 3]
    # model = ResNetShared(BasicBlock, [2, 2, 2, 2], **kwargs) # [3, 4, 6, 3]
    model = ResNetShared(Bottleneck, [2, 2, 2, 2], **kwargs) # [3, 4, 6, 3]
    return model

class Decoder(nn.Module):

    def __init__(self, in_planes, final_upsample_mode = 'nearest'):
        super(Decoder, self).__init__()

        self.conv1x1_1 = self._make_conv1x1_upsampled(in_planes[0], 64)
        self.conv1x1_2 = self._make_conv1x1_upsampled(in_planes[1], 64, 2)
        self.conv1x1_3 = self._make_conv1x1_upsampled(in_planes[2], 64, 4)
        self.conv1x1_4 = self._make_conv1x1_upsampled(in_planes[3], 64, 8)
        self.cv = nn.Conv2d(64 + 4*64, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv1x1_upsampled(self, inplanes, outplanes, scale_factor=None):
        if scale_factor:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, l):
        k = []
        k.append(l[0])
        x = self.conv1x1_1(l[1])
        k.append(self.conv1x1_1(l[1]))
        x = self.conv1x1_2(l[2])
        k.append(self.conv1x1_2(l[2]))
        x = self.conv1x1_3(l[3])
        k.append(self.conv1x1_3(l[3]))
        x = self.conv1x1_4(l[4])
        k.append(self.conv1x1_4(l[4]))
        cated = self.cv(torch.cat(k, 1))
        cated = self.sigm(cated)
        return cated

def DecoderShared(**kwargs):
    return Decoder([256, 512, 1024, 2048], **kwargs)


F_k = {}

def train_or_eval(data_loader, data_loader_w_norm, classifier, decoder, args, to_draw, version, train=False, optimizer=None, epoch=None):
    acc, acc_m, acc_m_bin = AverageMeter(), AverageMeter(), AverageMeter()
    backup_acc = []
    
    classifier_criterion = nn.CrossEntropyLoss().to(device)
    decoder.train() if train else decoder.eval()
    classifier.train() if train and not args.fixed_classifier else classifier.eval()
    
    for i, (input, target) in enumerate(data_loader):
        if train and i > len(data_loader) * args.pot:
            break

        input, target = input.to(device), target.to(device)
        with torch.set_grad_enabled(train and (not args.fixed_classifier)):
            output, layers = classifier(input)
            classifier_loss = classifier_criterion(output, target)

        batch_accuracy = accuracy(output.detach(), target, topk=(1,))[0].item()    
        backup_acc.append(str(batch_accuracy))                
        acc.update(batch_accuracy, input.size(0))

        if train and (not args.fixed_classifier):
            optimizer['classifier'].zero_grad()
            classifier_loss.backward()
            optimizer['classifier'].step()
            
            ## save classifier (needed only if previous iterations are used i.e. args.hp > 0)
            global F_k
            if args.hp > 0 and ((i % args.smf == -1 % args.smf) or len(F_k) < 1):
                print('Current iteration is saving, will be used in the future. ', end='', flush=True)
                if len(F_k) < args.f_size:
                    index = len(F_k) 
                else:
                    index = random.randint(0, len(F_k) - 1)
                state_dict = classifier.state_dict()
                F_k[index] = {}
                for p in state_dict:
                    F_k[index][p] = state_dict[p].cpu()
                print('There are {0} iterations stored.'.format(len(F_k)), flush=True)

         
        #line = create_line(args, version, batch_accuracy, masked_accuracy, bin_masked_accuracy)
        line = create_line(args, version, batch_accuracy, 0.0, 0.0)
        backup_acc.append(line)
        
    if not train:
        print(' * Prec@1 {acc.avg:.3f} Prec@1(M) {acc_m.avg:.3f} Prec@1(BM) {acc_m_bin.avg:.3f}'.format(
            acc=acc, acc_m=acc_m, acc_m_bin=acc_m_bin))

    update_backup_after_epoch(args, version, backup_acc)    
    
    to_draw[f"{version}_acc"].append(acc.avg)
    to_draw[f"{version}_acc_m"].append(acc_m.avg)
    to_draw[f"{version}_acc_m_bin"].append(acc_m_bin.avg)
    
    return {
        'acc':str(acc.avg),
        'acc_m':str(acc_m.avg),
        'acc_m_bin':str(acc_m_bin.avg),
    }


def get_to_draw_map():
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

def create_models(args):
    print("=> creating models...")
    
    classifier = resnet18shared(pretrained=True).to(device)
    decoder = DecoderShared(final_upsample_mode=args.upsample).to(device)

    optimizer = {}
    optimizer['classifier'] = torch.optim.SGD(classifier.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer['decoder'] = torch.optim.Adam(decoder.parameters(), args.lr_casme, weight_decay=args.weight_decay)
    
    return classifier, decoder, optimizer

def update_to_draw(results_by_epoch, to_draw):
    keys = ["train_acc", "test_hint_acc", "test_pure_acc", "train_acc_m", "test_hint_acc_m", "test_pure_acc_m", 
           "train_acc_m_bin", "test_hint_acc_m_bin", "test_pure_acc_m_bin"]
    for i, k in enumerate(keys):
        to_draw[k] = results_by_epoch[i]
    
def read_old_averaged_results(args, to_draw):
    with open(args.averaged_results, 'r') as backup:
        lines = backup.readlines()
        for line in lines:
            results_by_epoch = line.split(" ")
            results_by_epoch = [float(result) for result in results_by_epoch]
            update_to_draw(results_by_epoch, to_draw)
  
def restore(args, classifier, decoder, optimizer, to_draw):
    filename = 'xxx.chk'
    last_checkpoint_file = os.path.join(LAST_MODEL, filename)
    starting_epoch = 0
    #starting_epoch = load_checkpoint(optimizer, classifier, decoder, last_checkpoint_file)
    if starting_epoch != 0: 
        args.name = filename.split('.')[0]
    set_args(args)
    
    if starting_epoch != 0:
        read_old_averaged_results(args, to_draw)
        
    return starting_epoch


def read_one_backup_file(args, path, to_draw):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Epoch"):
                pass
                #insert new array
            else:
                results = line.split(" ")
                results = map(results, lambda x: float(x))
                
    assert("Dokoncz" == "To")
    
def convert_backup_strings_into_floats(args, to_draw):
    # prepare nine arrays to draw
    paths_to_restore = [args.details_acc, args.details_mask, args.details_bin_mask]
    for path in paths_to_restore:
        read_one_backup_file(args, path, to_draw)      


# MAIN FUNCTION
# ====================

# In[121]:


def main(logpath, lambda_r=10, epochs=300, prob=0.):
    args = arguments_setup(logpath)
    args.lambda_r = lambda_r
    args.epochs = epochs
    args.hint_prob = prob
    
    to_draw = get_to_draw_map()
    torch.backends.cudnn.deterministic=True
    #torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    classifier, decoder, optimizer = create_models(args)
    starting_epoch = restore(args, classifier, decoder, optimizer, to_draw)
    
    cudnn.benchmark = True
    train_loader, val_hint_loader, val_hint_loader_w_norm, val_pure_loader, val_pure_loader_w_norm = data_loading(args)
    best_acc, best_acc_pure, best_acc_mask = 0., 0., 0.
    barriers = [200, 320]
    
    for epoch in range(starting_epoch, args.epochs):
        print(f"Epoch {epoch}. Results {best_acc} {best_acc_pure} {best_acc_mask}")
        
        adjust_learning_rate(optimizer, 'classifier', barriers, epoch, args)
        adjust_learning_rate(optimizer, 'decoder', barriers, epoch, args)
        
        vers = ["train", "test_hint", "test_pure"]
        #append_epoch_to_backup_files(args, epoch) # Insert start of the new epoch
    
        tr_s = train_or_eval(train_loader, train_loader, classifier, decoder, args, to_draw, vers[0], True, optimizer, epoch)
        val_hint = train_or_eval(val_hint_loader, val_hint_loader_w_norm, classifier, decoder, args, to_draw, vers[1], epoch=epoch)
        val_pure = train_or_eval(val_pure_loader, val_pure_loader_w_norm, classifier, decoder, args, to_draw, vers[2], epoch=epoch)
        
        # if (epoch + 1) % args.freq_plot == 0:
        #     plot_all_accuracies(to_draw, args)
        
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
        
        logging_averaged(args, tr_s, val_hint, val_pure)
        
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
    main(logpath, lambda_r=12, epochs=400, prob=0.)