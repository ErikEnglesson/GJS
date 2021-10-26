import torch, os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import random
import json

import numpy as np
from PIL import Image


class ImageNetVal(datasets.ImageNet):
    def __init__(self, root, num_classes=50, split='val', **kwargs):
        super(ImageNetVal, self).__init__(root, split=split, **kwargs)

        # Extract only the images of the first num_classes classes
        self.samples = list() # This is part of the superclass of ImageNet
        for img, target in self.imgs:
            if target < num_classes:
                self.samples.append((img,target))

        # Update members of subclass too.
        self.imgs = self.samples
        self.targets = [target for img, target in self.samples]


class WebVision(data.Dataset):
    """
    The dataset consists of two main files:
    * train_filelist_google.txt are path-label pairs for training images
    * val_filelist.txt are path-label pairs for validation images

    Here, we only consider the mini version of WebVision version 1. See https://arxiv.org/abs/1712.05055

    There are X training images, Y validation images and Z test images.
    """

    def __init__(self, root, type='train', transforms=None, 
                 target_transform=None,
                 num_classes=50,
                 path_to_logits=None):
        
        assert type == 'train' or type == 'val'
        self.type = type
        self.root = root
        self.image_transforms = transforms
        self.target_transform = target_transform

        # Image path files for different dataset types
        self.image_paths_files = {'train': 'info/train_filelist_google.txt', 
                                  'val': 'info/val_filelist.txt'}

        # Number of samples for different dataset types
        self.num_samples = {'train': 980449, 
                            'val': 50000}
        self.num_classes = num_classes
        self.img_paths, self.path_to_label = self.load_paths_and_labels()
        self.num_samples[self.type] = len(self.img_paths)
        print(self.type, "has", self.num_samples[self.type],"examples!")
        self.img_paths_train = self.img_paths
        self.path_to_logits = path_to_logits
        if path_to_logits is not None and type == 'train':
            with open(path_to_logits) as f:
                data = json.load(f)
                print("Opening file")
            self.path_to_logits = data
                #for key in self.path_to_label.keys():
                #    self.path_to_label[key] = self.path_to_label[key][0:2]


        print("Number of examples per class:\n", np.bincount(list(self.path_to_label.values())))
        print("Number of classes in WebVision:", self.num_classes, "for", self.type)
        #self.label_to_name = self.load_labels_to_name()
        #print(self.label_to_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path = self.img_paths_train[index]
        label = self.path_to_label[path] 
        img = self.load_img(path)

        logits = self.path_to_logits[path] if self.path_to_logits is not None else None

        transformed_imgs = list()
        for transform in self.image_transforms:
            image = img.copy()
            image = transform(image)
            transformed_imgs.append(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        # TODO: Remove last label. This is not needed when separated into its own file.
        if self.type == 'train':
            if logits is not None:
                return transformed_imgs, label, logits
            else:
                return transformed_imgs, label, label
        else:
            return transformed_imgs[0], label

    def __len__(self):
        return self.num_samples[self.type]

    #def get_label_name(self, index):
    #    return self.label_to_name[index]

    def load_img(self, img_path):
        if self.type == 'val':
            path = self.root + 'val_images_256/' +  img_path
        else:
            path = self.root + img_path
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    #def load_labels_to_name(self):
    #    label_to_name, query_to_name = dict()
    #    path = self.root + 'info/queries_google.txt'
    #    with open(path, 'r') as file:
    #        for i, line in enumerate(file):
    #            if i > self.num_classes:
    #                continue

    #            query, name = line.split()
    #            query_to_name[int(label)] = name

        

    #    return label_to_name

    def load_paths_and_labels(self):
        path = self.root + self.image_paths_files[self.type]
        image_paths = [None] * self.num_samples[self.type]
        paths_to_labels = dict()
        with open(path, 'r') as file:
            if self.type == 'test':
                for i, line in enumerate(file):
                    image_paths[i] = line.replace('\n','')
            else:
                for i, line in enumerate(file):
                    path, label = line.split()

                    if int(label) >= self.num_classes:
                        continue

                    paths_to_labels[path] = int(label)
                    image_paths[i] = path
        
        image_paths = [path for path in image_paths if path is not None]
        return image_paths, paths_to_labels

    def generate_targets(self):
        targets = [None] * self.num_samples[self.type]
        for i, path in enumerate(self.img_paths_train):
            targets[i] = self.path_to_label[path]
        
        return targets


import numpy as np
import matplotlib.pyplot as plt
from RandAugment import RandAugment
from RandAugment.augmentations import *


def imshow(img, name):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(name)


if __name__ == "__main__":
    dataset_dir = '/local_storage/datasets/WebVision/'
    dataset_dir_imgnet = '/local_storage/datasets/imagenet/'

    transform_strong_aug = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    #transform_strong_aug.transforms.insert(0, RandAugment(1, 3))
    #transform_strong_aug.transforms.append(CutoutDefault(64))
    
    #dataset = WebVision(dataset_dir, type='train', transforms=[transform_strong_aug])
    dataset = ImageNetVal(dataset_dir_imgnet, transform=transform_strong_aug)
    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)

    dataiter = iter(dataloader)
    #images, labels, indices = dataiter.next()
    images, labels = dataiter.next()
    
    for i in range(16):
        label = labels[i].item()
        #name = dataset.get_label_name(label)
        #print(f'label: {label}, name: {name}')
        print(f'label: {label}')

    imshow(torchvision.utils.make_grid(images),'test0.png')
    #imshow(torchvision.utils.make_grid(images[1]),'test1.png')
