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


from models import *
from models import resnet26

from datetime import datetime

import tqdm

# from figure_utils import create_figures
import datasets.cifar as cifar
import datasets.webvision as webvision
from datasets.transforms import get_webvision_transforms

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

def get_network(num_classes, args):
    if args.arch == 'resnet50':
        net = torchmodels.resnet50(pretrained=not args.scratch)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)
    elif args.arch == 'densenet':
        net = torchmodels.densenet121(pretrained=not args.scratch, drop_rate=0.2)
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(num_ftrs, num_classes)
    elif args.arch == 'preactresnet':
        net = PreActResNet34(num_classes)
    elif args.arch == 'resnet26':
        model_factory = resnet26.__dict__['cifar_shakeshake26']
        model_params = dict(pretrained=False, num_classes=num_classes)
        net = model_factory(**model_params)
    else:
        assert(False)

    return net

def set_drop_lr(epoch, lr_warmup, lr_start, drop1=40, drop2=80, start_epoch=0, drop_factor=0.1):
    assert drop1 < drop2

    if epoch > start_epoch and epoch < drop1:
        return lr_start / lr_warmup
    elif epoch >= drop1 and epoch < drop2:
        return drop_factor * (lr_start / lr_warmup)
    elif epoch >= drop2:
        return drop_factor * drop_factor * (lr_start / lr_warmup)
    else:
        return 1.0


def weight_decay_loss(net, wd_factor, skip_list=()):
    wd = torch.tensor(0.0).cuda()
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            continue

        wd += (param**2).sum()

    return wd_factor * wd

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_constant_lr(epoch, lr_warmup, lr_start, start_epoch=0):
    if epoch > start_epoch:
        return lr_start / lr_warmup
    else:
        return 1.0

def set_seeds(seed):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

def main(args):

    set_seeds(args.seed)

    assert(torch.cuda.is_available())
    device = torch.device('cuda')

    # -- TensorBoard --
    time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    random_hash = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    log_dir = args.run_dir + f"arch{args.arch}_lrschedule{args.lr_schedule}_loss{args.loss}_losstype{args.dissect_js}_a{args.loss_alpha}_b{args.loss_beta}_wd{args.weightdecay}_augs{args.augs}_bspg{args.bs_pg}_e{args.epochs}_seed{args.seed}_lrpg{args.lr_pg}_lrd1{args.lr_drop1}_lrd2{args.lr_drop2}_percent{args.percent}_asym{int(args.asym)}_mo{args.momentum}_N{args.N}_M{args.M}_{time}_{random_hash}/"

    writer = SummaryWriter(log_dir=log_dir)
    print("Log dir: ", log_dir)

    # Store this file in the log directory
    subprocess.run(["cp", sys.argv[0], log_dir])
    subprocess.run(["cp", 'losses.py', log_dir])

    # -- Data Set ---
    classes = {'cifar10': 10, 'cifar100': 100, 'webvision': args.webvision_classes}
    num_classes = classes[args.dataset]
    
    # Set seeds to get same synthetic datasets
    set_seeds(args.seed)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        trainset, valset = cifar.get_cifar(args.data_dir, args, train=not args.test, 
                                           download=True, num_classes=num_classes)
        print("Targets for training set:", trainset.targets_noisy[:5])
    elif args.dataset == 'webvision':
        transform_no_aug, transform_weak_aug, transform_strong_aug = \
        get_webvision_transforms(args)

        print("Creating transformations for: ", args.augs)
        image_transforms = list()
        for transform in args.augs:
            if transform == 'w':
                image_transforms.append(transform_weak_aug)
            elif transform == 's':
                image_transforms.append(transform_strong_aug)
            elif transform == 'n':
                image_transforms.append(transform_no_aug)
            else:
                assert False

        trainset = webvision.WebVision(args.data_dir, 
                                         type='train', 
                                         transforms=image_transforms,
                                         num_classes=num_classes)
        dataset_type = 'test' if args.test else 'val'
        if dataset_type == 'val':
            valset = webvision.WebVision(args.data_dir, 
                                         type='val', 
                                         transforms=[transform_no_aug],
                                         num_classes=num_classes)
        else:
            imagenet_path = '/local_storage/datasets/imagenet/'
            valset = webvision.ImageNetVal(imagenet_path, 
                                           transform=transform_no_aug,
                                           num_classes=num_classes)


    if args.eval_consistency:
        assert args.dataset == 'cifar10' or args.dataset == 'cifar100'
        set_seeds(args.seed)
        consistencyset, _ = cifar.get_cifar(args.data_dir, args, train=True, 
                                        download=True, num_classes=num_classes,
                                        augs='n' + args.augs[0])
        consistencyloader = data.DataLoader(consistencyset, batch_size=args.bs_pg, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.num_workers)

    # -- Data Loaders --
    trainloader = data.DataLoader(trainset, batch_size=args.bs_pg, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.num_workers)
    valloader = data.DataLoader(valset, batch_size=args.bs_pg, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    dataloaders = {'train': trainloader, 'val': valloader}
    dataset_sizes = {'train': len(trainset), 'val': len(valset)}
    print("Dataset sizes: ", dataset_sizes, ", Classes:", num_classes)

    # -- Networks --
    # Set seeds to get same weight initialization
    set_seeds(args.seed)
    pg = get_network(num_classes, args)
    pg = pg.to(device)

    # -- Optimizer --
    optimizer = optim.SGD(pg.parameters(), 
                          lr=args.lr_pg, 
                          momentum=args.momentum, 
                          weight_decay=args.weightdecay, 
                          nesterov=args.nesterov)

    # -- Learning Rate Scheduling --
    if args.lr_schedule == 'drop':
        lambda_fn = lambda epoch: set_drop_lr(epoch, args.lr_pg, args.lr_pg,
                                              args.lr_drop1, args.lr_drop2)
    elif args.lr_schedule == 'cos':
        lambda_fn = lambda epoch: np.cos(args.cosw * np.pi * epoch / args.epochs)
  
    if args.lr_schedule == 'stepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.stepLRfactor)
    else:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)

    # -- Loss --
    criterion = losses.get_criterion(num_classes, args)

    # -- Train Network --
    acc_best = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(dataloaders, pg, device, optimizer, criterion, epoch, writer, log_dir, dataset_sizes, num_classes, args)
        scheduler.step()

        acc_best, acc = val(dataloaders, pg, device, epoch, writer, args.test, acc_best)
          
        if args.eval_consistency:
            eval_consistency(consistencyloader, pg, device, epoch, writer)

    writer.close()
    print("Best Acc:", acc_best, "Last Acc:", acc)


def train_one_epoch(dataloaders, net, device, optimizer, criterion, epoch, writer, log_dir, dataset_sizes, num_classes, args):
              
    net.train()
    batch_bar = tqdm.tqdm(dataloaders['train'], desc='Batch')
    start_time = time.time()

    meter_loss, meter_noise = AverageMeter(), AverageMeter()

    # Create three meters to keep track of accuracy for samples with true labels, noisy labels and all.
    acc_meters = list()
    for _ in args.augs:
        acc_meters.append({'true': AverageMeter(), 'noisy': AverageMeter(), 'all': AverageMeter()})
    
    for batch_idx, (input_list, labels_noisy, labels_true) in enumerate(batch_bar):
        labels_noisy, labels_true = labels_noisy.to(device), labels_true.to(device)

        optimizer.zero_grad()

        # Evaluate network on all augmented batches
        train_logits = list()
        if args.onebatch:
            inputs_all = torch.cat(input_list, 0).to(device)
            logits_all = net(inputs_all)
            train_logits = list(torch.split(logits_all, input_list[0].size(0)))
        else:
            for image_batch in input_list:
                batch = image_batch.to(device)
                train_logits.append(net(batch))

        pred = train_logits[0] if len(train_logits) == 1 else train_logits
        loss = criterion(pred, labels_noisy)

        loss.backward()
        optimizer.step()
        
        # Logging
        num_samples = labels_noisy.shape[0]
        meter_loss.update(loss, num_samples)
        acc_tl_nl = (labels_noisy == labels_true).float().mean()
        meter_noise.update(acc_tl_nl, num_samples)
        assert len(train_logits) == len(acc_meters)
        for i, acc_meter in enumerate(acc_meters):
            predictions = train_logits[i].argmax(dim=1)
            mask_true = labels_true == labels_noisy
            acc_true = (predictions[mask_true] == labels_true[mask_true]).float().mean()
            acc_noisy = (predictions[~mask_true] == labels_true[~mask_true]).float().mean()
            acc_all = (predictions == labels_true).float().mean()
            acc_meter['true'].update(acc_true, mask_true.int().sum())
            acc_meter['noisy'].update(acc_noisy, num_samples-mask_true.int().sum())
            acc_meter['all'].update(acc_all, num_samples)

                
    # Save checkpoints to disk
    torch.save(net.state_dict(), log_dir + 'pg_checkpoint.pt')

    # Logging & TensorBoard 
    epoch_time = time.time() - start_time 

    train_loss = meter_loss.get_average()
    writer.add_scalar("Train/Loss", train_loss, epoch)
    noise_level = meter_noise.get_average()
    writer.add_scalar("Train/Acc-TLvsNL", noise_level, epoch)
    for i, acc_meter in enumerate(acc_meters):
        acc_true = acc_meter['true'].get_average()
        acc_noisy = acc_meter['noisy'].get_average()
        acc_all = acc_meter['all'].get_average()
        writer.add_scalar(f"Train/Acc-TrueSamples-{args.augs[i]}-{i}", acc_true, epoch)
        writer.add_scalar(f"Train/Acc-NoisySamples-{args.augs[i]}-{i}", acc_noisy, epoch)
        writer.add_scalar(f"Train/Acc-AllSamples-{args.augs[i]}-{i}", acc_all, epoch)

    writer.add_scalar("Other/learning_rate", get_lr(optimizer), epoch) 
    weight_norm = weight_decay_loss(net, 1.0).detach().cpu().numpy()
    writer.add_scalar("Other/weight_l2norm", weight_norm, epoch)

    acc_all = acc_meters[0]['all'].get_average()
    print(f"[Epoch: {epoch:.2f}] | Time: {epoch_time:.2f}] | Train TL-PL Acc: {acc_all:.3f}")


def val(dataloaders, net, device, epoch, writer, is_test, acc_best):
    net.eval()
    meter_loss, meter_acc, meter_acc_noisy = AverageMeter(), AverageMeter(), AverageMeter()

    start_time = time.time()
    for data in dataloaders['val']:

        if len(data)==2:
            (inputs, labels) = data    
            inputs, labels_noisy, labels = inputs.to(device), None, labels.to(device)
        else:
            (input_list, labels_noisy, labels_true) = data
            inputs = input_list[0]
            inputs, labels_noisy, labels = inputs.to(device), labels_noisy.to(device), labels_true.to(device)

        with torch.set_grad_enabled(False):
            val_logits = net(inputs)
            val_loss = F.cross_entropy(val_logits, labels)
 
            meter_loss.update(val_loss, labels.shape[0])
            meter_acc.update((val_logits.argmax(dim=1) == labels).float().mean(), labels.shape[0])
            if labels_noisy is not None:
                meter_acc_noisy.update((val_logits.argmax(dim=1) == labels_noisy).float().mean(), labels_noisy.shape[0])

    loss, acc = meter_loss.get_average(), meter_acc.get_average() 
    if labels_noisy is not None:
        acc_noisy = meter_acc_noisy.get_average()
    eval_time = time.time() - start_time

    acc_best = max(acc, acc_best)
    data = 'Test' if is_test else 'Validation'
    print(f'[Epoch: {epoch:.2f}] {data} CE: {loss:.3f} | {data} Acc: {acc:.3f} | Best {data} Acc: {acc_best:.3f} ')
    writer.add_scalar(f'{data}/CE-TLvsPL', loss, epoch)
    writer.add_scalar(f'{data}/Acc-TLvsPL', acc, epoch)
    if labels_noisy is not None:
        writer.add_scalar(f'{data}/Acc-NLvsPL', acc_noisy, epoch)

    print(f'{data} evaluation time: ', eval_time)

    return acc_best, acc


def eval_consistency(dataloader, net, device, epoch, writer):
    net.eval()
    batch_bar = tqdm.tqdm(dataloader, desc='Batch')

    acc_meter = {'true': AverageMeter(), 'noisy': AverageMeter(), 'all': AverageMeter()}

    start_time = time.time()
    for batch_idx, (input_list, labels_noisy, labels_true) in enumerate(batch_bar):
        labels_noisy, labels_true = labels_noisy.to(device), labels_true.to(device)
        num_samples = labels_noisy.shape[0]

        with torch.set_grad_enabled(False):
            # Evaluate network on all augmented batches
            train_logits = list()
            for image_batch in input_list:
                batch = image_batch.to(device)
                train_logits.append(net(batch))


            # -- Evaluate consistency metric --
            predictions1 = train_logits[0].argmax(dim=1)
            predictions2 = train_logits[1].argmax(dim=1)

            mask_true = labels_true == labels_noisy
            acc_true =  (predictions1[ mask_true] == predictions2[ mask_true]).float().mean()
            acc_noisy = (predictions1[~mask_true] == predictions2[~mask_true]).float().mean()
            acc_all = (predictions1 == predictions2).float().mean()
            acc_meter['true'].update(acc_true, mask_true.int().sum())
            acc_meter['noisy'].update(acc_noisy, num_samples-mask_true.int().sum())
            acc_meter['all'].update(acc_all, num_samples)

    acc_true, acc_noisy, acc_all = acc_meter['true'].get_average(), acc_meter['noisy'].get_average(), acc_meter['all'].get_average()
    eval_time = time.time() - start_time

    writer.add_scalar(f'Train/Acc-Consistency-True', acc_true, epoch)
    writer.add_scalar(f'Train/Acc-Consistency-Noisy', acc_noisy, epoch)
    writer.add_scalar(f'Train/Acc-Consistency-All', acc_all, epoch)
    print(f'Consistency evaluation time: ', eval_time)


def parse_args():
    argparser = configargparse.ArgumentParser()

    argparser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
    argparser.add_argument('--seed', type=int, help='random seed', default=0)

    # Logging
    argparser.add_argument('-rd', '--run_dir', type=str, help='Path to runs directory', default='runs/')

    # Network
    argparser.add_argument('-a', '--arch', type=str, choices=['resnet26', 'preactresnet', 'resnet50', 'densenet', 'wideresnet'], default='preactresnet')
    argparser.add_argument('-s', '--scratch', help='train from scratch instead of fine tuning.', action='store_true')

    # Noise
    argparser.add_argument('--percent', type=float, help='Percent of noise in labels', default=0.0)
    argparser.add_argument('--asym', help='Assymetric noise', action='store_true')

    # Optimizer
    argparser.add_argument('-mo', '--momentum', help='momentum', type=float, default=0.9)
    argparser.add_argument('-wd', '--weightdecay', type=float, help='Weight decay factor for PG', default=1e-4)
    argparser.add_argument('-bspg', '--bs_pg', type=int, help='Training batch size', default=128)
    argparser.add_argument('-e', '--epochs', type=int, help='Number of training epochs', default=300)
    argparser.add_argument('--nesterov', help='Number of training epochs', action='store_true')

    # Data
    argparser.add_argument('-dd', '--data_dir', type=str, help='Path to dataset', default='../tensorpack_data/cifar10_data/')
    argparser.add_argument('--num_workers', '-nw', help='number of workers for data loaders', type=int, default=4)
    argparser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'webvision'], help='which dataset to use')
    argparser.add_argument('--augs', help='string of n(no), w(weak) and s(strong) augs', type=str, default='s')
    argparser.add_argument('--augmentation', type=str, help='which type of strong augmentation to use', choices=['rand'], default='rand')
    argparser.add_argument('--N', type=int, help='HP for RandAugment', default=3)
    argparser.add_argument('--M', type=int, help='HP for RandAugment', default=9)
    argparser.add_argument('--cutout', type=int, help='length used for cutout, 0 disables it', default=16)
    argparser.add_argument('--test', help='Train on full training set and evaluate on the test set.', action='store_true')
    argparser.add_argument('--webvision_classes', type=int, help='Number of classes', default=50)

    # Learning rate 
    argparser.add_argument('-lrpg', '--lr_pg', type=float, help='learning rate for prediction generator', default=0.1)
    argparser.add_argument('--lr_schedule', type=str, choices=['cos', 'drop', 'stepLR'], default='cos')
    argparser.add_argument('--cosw', help='multiplicative factor in the cos lr scheduling', type=float, default=0.4375)
    argparser.add_argument('--lr_drop1', '-ld1', type=int, help='epoch for first lr drop', default=40)
    argparser.add_argument('--lr_drop2', '-ld2', type=int, help='epoch for second lr drop', default=80)
    argparser.add_argument('--stepLRfactor', type=float, default=0.97)
    argparser.add_argument('--cos_epochs', type=int, help='used for cos_continue lr', default=300)

    # Loss function
    argparser.add_argument('--loss', choices=['JSDissect', 'CE', 'NCE+RCE', 'SCE', 'LS', 'MAE', 'GCE', 'JSWC', 'JSWCS', 'JSNoConsistency', 'bootstrap'])
    argparser.add_argument('--loss_alpha', default=1.0, type=float)
    argparser.add_argument('--loss_beta', default=1.0, type=float)
    argparser.add_argument('--q', default=0.7, type=float)
    argparser.add_argument('--js_weights', help='First weight is for label, the next are in the order of "augs"', type=str, default='0.5 0.5')
    argparser.add_argument('--onebatch', help='Evaluate images as a single batch.', action='store_true')

    # JS specific
    argparser.add_argument('--dissect_js', type=str, choices=['a','b','c','d','e','f', 'as','bs','cs','ds','es','fs', 'g','h', 'i']) # Defaults to None

    # Evaluation
    argparser.add_argument('--eval_consistency', help='Check consistency of noisy vs noise free training examples.', action='store_true')

    return argparser.parse_args()


if __name__ == '__main__':
    print("This is the name of the script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    print("The arguments are: " , str(sys.argv))
    args = parse_args()
    print(args)
    
    main(args)
