
import torchvision.transforms as transforms

from RandAugment import RandAugment
from RandAugment.augmentations import *

def get_cifar10_transforms(args):

    transform_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_weak_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # TODO: Remove auto augment when everything works as it should.
    if args.augmentation == 'rand':    
        transform_strong_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # Add RandAugment with N, M(hyperparameter)
        transform_strong_aug.transforms.insert(0, RandAugment(args.N, args.M))
    else:
        print("No strong augmentation given, using weak.")
        transform_strong_aug = transform_weak_aug
    
    if args.cutout > 0:
        transform_strong_aug.transforms.append(CutoutDefault(args.cutout))

    return transform_no_aug, transform_weak_aug, transform_strong_aug

def get_cifar100_transforms(args):
    transform_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    transform_weak_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),        
    ])

    if args.augmentation == 'rand':    
        transform_strong_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),        
        ])
        # Add RandAugment with N, M(hyperparameter)
        # 2, 14 seemed to work well for WRN28-10
        transform_strong_aug.transforms.insert(0, RandAugment(args.N, args.M))
    else:
        print("No strong augmentation given, using weak.")
        transform_strong_aug = transform_weak_aug

    if args.cutout > 0:
        transform_strong_aug.transforms.append(CutoutDefault(args.cutout))

    return transform_no_aug, transform_weak_aug, transform_strong_aug


def get_webvision_transforms(args):

    if args.arch == 'IResNetV2':
        transform_no_aug = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 

        transform_weak_aug = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ]) 

        transform_strong_aug = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

        # Add RandAugment with N, M(hyperparameter)
        transform_strong_aug.transforms.insert(0, RandAugment(args.N, args.M))
    
        if args.cutout > 0:
            transform_strong_aug.transforms.append(CutoutDefault(args.cutout))
            
    else:

        transform_no_aug = transforms.Compose([transforms.Resize(256), # 256 vs 74
                                                transforms.CenterCrop(224), # 224 vs 64
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                        ])

        transform_weak_aug = transforms.Compose([
            transforms.RandomResizedCrop(224), #224 vs 64
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4,
                                contrast=0.4,
                                saturation=0.4,
                                hue=0.2),
            transforms.ToTensor(),                
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ]) 

        transform_strong_aug = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

        # Add RandAugment with N, M(hyperparameter)
        transform_strong_aug.transforms.insert(0, RandAugment(args.N, args.M))
    
        if args.cutout > 0:
            transform_strong_aug.transforms.append(CutoutDefault(args.cutout))
            
    return transform_no_aug, transform_weak_aug, transform_strong_aug
