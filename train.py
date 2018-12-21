from __future__ import print_function, division

import torch
import argparse
import json 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from functools import partial 
#import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import random
import copy
import pdb 
from tqdm import tqdm
from utils import * 
import torchvision.utils as vutils
from drn import *

device = [0]

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--workers', default=10, type=int, 
                        help='number of data loading workers')
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='batch size')
    parser.add_argument('--DRN', default=True, type=bool, 
                        help='if use DRN')

    args = parser.parse_args()
    return args

def transfrom_data():
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
    }

    return data_transforms

def load_data(batch_size, num_workers):
    print("Start loading data")
    image_datasets = {x: datasets.FashionMNIST(root='./', train=bool(x=='train'), transform=transfrom_data()[x], download=True) \
                        for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) \
                    for x in ['train', 'test']}
    class_num = 10
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print("Dataset sizes: Train {} Test {}".format(dataset_sizes['train'], dataset_sizes['test']))

    return dataloaders, class_num, dataset_sizes

def load_model_resnet(class_num):
    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_num)

    model = torch.nn.DataParallel(model.cuda(), device_ids=device)

    return model

def load_model_drn(class_num):
    model = drn_a_50(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_num)

    model = torch.nn.DataParallel(model.cuda(), device_ids=device)

    return model 

def train_model(dataloaders, model, dataset_sizes, criterion, optimizer, scheduler, num_epochs, save_dir, f):
    since = time.time()
    
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_val_top1_acc = 0.0
    best_val_epoch = -1 
    final_val_top5_acc = 0.0
    best_test_top1_acc = 0.0
    best_test_epoch = -1
    final_test_top5_acc = 0.0 

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            top1_running_corrects = 0
            top5_running_corrects = 0

            it = tqdm(range(len(dataloaders[phase])), desc="Epoch {}/{}, Split {}".format(epoch, num_epochs - 1, phase), ncols=0)
            data_iter = iter(dataloaders[phase])
            for niter in it:
                inputs, labels = data_iter.next()
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    if phase == 'train':

                        outputs = model(inputs.cuda())
                        prec1, prec5 = accuracy(outputs, labels.cuda(), topk=(1,5))
                        loss = criterion(outputs, labels.cuda())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                    else:
                        outputs = model(inputs.cuda())
                        prec1, prec5 = accuracy(outputs, labels.cuda(), topk=(1,5))

                training_loss = loss.item()
                running_loss += loss.item() * inputs.size(0)
                top1_running_corrects += prec1[0]
                top5_running_corrects += prec5[0]

            epoch_loss = running_loss / dataset_sizes[phase]
            top1_epoch_acc = float(top1_running_corrects) / dataset_sizes[phase]
            top5_epoch_acc = float(top5_running_corrects) / dataset_sizes[phase]
            print('{} Epoch Loss: {:.6f} Epoch top1 Acc: {:.6f} Epoch top5 Acc: {:.6f}\n'.format(phase, epoch_loss, top1_epoch_acc, top5_epoch_acc))
            with open(epoch_trace_f_dir, "a") as f:
                lr = optimizer.param_groups[0]['lr']
                f.write("{},{},{},{:e},{:e},{:e}\n".format(epoch,phase,lr,epoch_loss,top1_epoch_acc,top5_epoch_acc))

            if phase == 'test' and top1_epoch_acc > best_test_top1_acc:
                print("Top1 test Acc improve from {:6f} --> {:6f}".format(best_test_top1_acc, top1_epoch_acc))
                best_test_top1_acc = top1_epoch_acc
                final_test_top5_acc = top5_epoch_acc
                best_test_epoch = epoch 
                save_f_dir = os.path.join(save_dir, "best_test_model.ft")
                print("Saving best test model into {}...".format(save_f_dir))
                torch.save(model.state_dict(), save_f_dir)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best top1 val Acc: {:6f}'.format(best_test_top1_acc))
    print('Final top5 val Acc: {:6f}'.format(final_test_top5_acc))
    print('Best test model is saved at epoch # {}'.format(best_test_epoch))


if __name__=="__main__":

    args = parse_args()
	
    dataloaders, class_names, dataset_sizes = load_data(args.batch_size, args.workers)
    if args.DRN:
        model= load_model_drn(class_names)
        name = 'drn'
    else:
        model= load_model_resnet(class_names)
        name = 'resnet'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    steps = 20
    scheduler = lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)
    num_epochs = 40

    save_dir = './outputs/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    epoch_trace_f_dir = os.path.join(save_dir, "trace_" + name + ".csv")
    with open(epoch_trace_f_dir, "w") as f:
        f.write("epoch,split,lr,loss,top1_acc,top5_acc\n")

    train_model(dataloaders, model, dataset_sizes, criterion, optimizer, scheduler, num_epochs, save_dir, f)

    
