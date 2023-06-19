import numpy
import os
import cv2
import torch

ann_path = 'tiny-imagenet-200/val/val_annotations.txt'
dirs_path = 'tiny-imagenet-200/wnids.txt'

fo = open(dirs_path, 'r+')
dirs = fo.readlines()
dirs = [fold.strip() for fold in dirs]
parent_path = 'tiny-imagenet-200/val'
new_val_folder = 'tiny-imagenet-200/val_new'

folders_content = {}
all_dirs = []

for d in dirs: 
    folders_content[d] = {}
    all_dirs.append(d)
    path = os.path.join(new_val_folder, d)
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'images'))

annotations = open(ann_path, 'r+')
lines = annotations.readlines()
lines = [line.strip() for line in lines]
print(lines[:10])
dir_names_for_images = [line.split(' ')[0].split('\t')[1] for line in lines]
print(dir_names_for_images[:10])

new_val_folder = 'tiny-imagenet-200/val_new'
# files = os.listdir(os.path.join(parent_path, 'images'))
files = [f'val_{i}.JPEG' for i in range(10000)]

for i, f in enumerate(files):
    # img = cv2.imread(os.path.join(fold_path, f), 1)
    # img = torch.FloatTensor(img).permute(2, 0, 1)    
    old_path = os.path.join(parent_path, f"images/{f}")
    label_path = dir_names_for_images[i]
    new_path = os.path.join(os.path.join(new_val_folder, label_path), f'images/{f}')
    if i<10:
        print(f"Next file {old_path} => {new_path}")
    os.rename(old_path, new_path)