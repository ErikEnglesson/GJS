import sys
import subprocess
import configargparse
from configparser import ConfigParser
import time
import typing
import random
import string
import yaml

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as torchmodels

import torch.utils.data as data
import torchvision.transforms as transforms


from models import *
from models import resnet26
from models import wideresnet
from models import preact_resnet_v2

from datetime import datetime

import tqdm

# from figure_utils import create_figures
import datasets.cifar as cifar
import datasets.clothing1m as clothing1m
import datasets.webvision as webvision
from datasets.transforms import get_clothing1m_transforms
from datasets.transforms import get_webvision_transforms

from warmup_scheduler import GradualWarmupScheduler

import losses

class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def get_average(self):
        if isinstance(self.average, torch.Tensor):
            return float(self.average.cpu().detach())
        return self.average

    def update(self, value, num):
        self.value = value
        self.sum += value * num
        self.count += num
        self.average = self.sum / self.count

    def __repr__(self):
        return f"{self.get_average():.4f}"


def get_network(num_classes):
   
    net = torchmodels.resnet50(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    
    return net

def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate_runs(models_paths, dataloaders, device, is_test, num_classes=50, arch='resnet50'):
    accs_top1 = np.zeros(len(models_paths))
    accs_top5 = np.zeros(len(models_paths))
    assert(len(models_paths) == 3)
    for i, model_path in enumerate(models_paths):
        set_seeds(42)
        pg = get_network(num_classes)

        pg.load_state_dict(torch.load(model_path))
        pg.eval()
        pg = pg.to(device)

        accs_top1[i], accs_top5[i] = val(dataloaders, pg, device, is_test)

    accs = {'top1': (accs_top1, np.mean(accs_top1), np.std(accs_top1)),
            'top5': (accs_top5, np.mean(accs_top5), np.std(accs_top5))}

    return accs

def main():

    assert(torch.cuda.is_available())
    device = torch.device('cuda')

    # -- Data Set ---
    num_classes = 50
    
    set_seeds(42)

    transform_no_aug = transforms.Compose([transforms.Resize(256), 
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                    ])

    webvision_path = '/local_storage/datasets/WebVision/'
    valset = webvision.WebVision(webvision_path, 
                                    type='val', 
                                    transforms=[transform_no_aug],
                                    num_classes=num_classes)
    
    imagenet_path = '/local_storage/datasets/imagenet/'
    testset = webvision.ImageNetVal(imagenet_path, 
                                    transform=transform_no_aug,
                                    num_classes=num_classes)

    # -- Data Loaders --
    valloader = data.DataLoader(valset, batch_size=64, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)

    dataloaders = {'val': valloader, 'test': testloader}
    dataset_sizes = {'val': len(valset), 'test': len(testset)}
    print("Dataset sizes: ", dataset_sizes, ", Classes:", num_classes)


    # -- Networks --
    paths_ce = ('runs/webvision/archresnet50_lrschedulestepLR_lossCE_a1.0_b1.0_wd0.0001_augsw_bspg64_e300_seed0_lrpg0.4_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-09-25_06-56_9G05/pg_checkpoint.pt', 
                'runs/webvision/archresnet50_lrschedulestepLR_lossCE_losstypeNone_a1.0_b1.0_wd0.0001_augsw_bspg64_e300_seed1_lrpg0.4_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-11-02_19-13_BF3I/pg_checkpoint.pt', 
                'runs/webvision/archresnet50_lrschedulestepLR_lossCE_losstypeNone_a1.0_b1.0_wd0.0001_augsw_bspg64_e300_seed2_lrpg0.4_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-11-02_19-13_AZTD/pg_checkpoint.pt')
 
    paths_js = ('runs/webvision/archresnet50_lrschedulestepLR_lossJSWC_a0.1_b0.9_wd0.0001_augsw_bspg64_e300_seed0_lrpg0.2_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-09-23_21-04_AFIK/pg_checkpoint.pt', 
                'runs/webvision/archresnet50_lrschedulestepLR_lossJSWC_losstypeNone_a0.1_b0.9_wd0.0001_augsw_bspg64_e300_seed1_lrpg0.2_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-11-03_22-47_8YKA/pg_checkpoint.pt', 
                'runs/webvision/archresnet50_lrschedulestepLR_lossJSWC_losstypeNone_a0.1_b0.9_wd0.0001_augsw_bspg64_e300_seed2_lrpg0.2_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-11-03_22-49_8QAY/pg_checkpoint.pt')

    paths_gjs = ('runs/webvision/archresnet50_lrschedulestepLR_lossJSWC_a0.1_b0.45_wd0.0001_augsww_bspg32_e300_seed0_lrpg0.1_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-09-25_03-22_CJTR/pg_checkpoint.pt', 
                'runs/webvision/archresnet50_lrschedulestepLR_lossJSWC_losstypeNone_a0.1_b0.45_wd0.0001_augsww_bspg32_e300_seed1_lrpg0.1_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-11-02_19-13_WK51/pg_checkpoint.pt', 
                'runs/webvision/archresnet50_lrschedulestepLR_lossJSWC_losstypeNone_a0.1_b0.45_wd0.0001_augsww_bspg32_e300_seed2_lrpg0.1_lrd140_lrd2500_percent0.0_mo0.9_N1_M3_size265664_2020-11-02_19-13_OFRN/pg_checkpoint.pt')

    paths_per_loss = {'ce': paths_ce,
                      'js': paths_js,
                      'gjs': paths_gjs}

    accs_per_loss_val, accs_per_loss_test = dict(), dict()
    for is_test in [False, True]:
        for loss in paths_per_loss.keys():
            paths = paths_per_loss[loss]
            acc = evaluate_runs(paths, dataloaders, device, is_test, num_classes, 'resnet50')
            #print(f"Acc for {loss}: {acc}")

            if is_test:
                accs_per_loss_test[loss] = acc
            else:
                accs_per_loss_val[loss] = acc

    print("Val:", accs_per_loss_val)
    print("Test:", accs_per_loss_test)
    print("---VALIDATION---")
    for key, item in accs_per_loss_val.items(): 
        print(f"{key} & {np.mean(item['top1'][0]):.2f} \pm {np.std(item['top1'][0]):.2f} & {np.mean(item['top5'][0]):.2f} \pm {np.std(item['top5'][0]):.2f}") 

    print("---TEST---")
    for key, item in accs_per_loss_test.items(): 
        print(f"{key} & {np.mean(item['top1'][0]):.2f} \pm {np.std(item['top1'][0]):.2f} & {np.mean(item['top5'][0]):.2f} \pm {np.std(item['top5'][0]):.2f}")  


def val(dataloaders, net, device, is_test):
    net.eval()
    meter_loss, meter_acc1, meter_acc5 =  AverageMeter(), AverageMeter(), AverageMeter()

    start_time = time.time()
    for data in dataloaders['test' if is_test else 'val']:

        (inputs, labels) = data    
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            val_logits = net(inputs)
            val_loss = F.cross_entropy(val_logits, labels)
 
            meter_loss.update(val_loss, labels.shape[0])
            a1 = (val_logits.argmax(dim=1) == labels).float().mean()
            meter_acc1.update(a1, labels.shape[0])
            a12 = (torch.topk(val_logits, 1, dim=1)[1] == labels.view(-1,1)).float().mean()
            assert a1 == a12
            a5 = (torch.topk(val_logits, 5, dim=1)[1] == labels.view(-1,1).expand(-1,5)).float().sum(dim=1).mean()
            meter_acc5.update(a5, labels.shape[0])

    loss, acc1, acc5 = meter_loss.get_average(), meter_acc1.get_average(), meter_acc5.get_average()
    eval_time = time.time() - start_time

    data = 'Test' if is_test else 'Validation'
    print("Loss:", loss)
    print("Acc1:", acc1, "Acc5:", acc5)
    print(f'{data} evaluation time: ', eval_time)

    return acc1*100.0, acc5*100.0



if __name__ == '__main__':    
    main()


