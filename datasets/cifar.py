# Modifications to:
# https://github.com/YU1ut/JointOptimization/blob/master/dataset/cifar10.py
import numpy as np
from PIL import Image

import torchvision

from datasets.transforms import get_cifar10_transforms, get_cifar100_transforms 

def get_cifar(root, args, train=True,
                download=False,
                num_classes=10,
                augs=None):
    
    is_cifar10 = num_classes == 10
    assert num_classes == 10 or num_classes == 100
    
    print("Creating transformations for: ", args.augs)
    tf_no, tf_weak, tf_strong = \
    get_cifar10_transforms(args) if is_cifar10 else get_cifar100_transforms(args)

    transforms = list()
    augs = args.augs if augs is None else augs
    for transform in augs:
        if transform == 'w':
            transforms.append(tf_weak)
        elif transform == 's':
            transforms.append(tf_strong)
        elif transform == 'n':
            transforms.append(tf_no)
        else:
            assert False

    if is_cifar10:
        base_dataset = torchvision.datasets.CIFAR10(root, train=True,
                                                    download=download)
        train_idxs, val_idxs = train_val_split(base_dataset.targets,
                                               num_classes,
                                               train=train)

        train_dataset = CIFAR10_train(root, train_idxs, args.percent,
                                      train=True, transforms=transforms)

        if train: 
            val_dataset = CIFAR10_train(root, val_idxs, args.percent,
                                        train=True, transforms=[tf_no])
        else:
            val_dataset = CIFAR10_val(root, val_idxs, train=False,
                                      transform=tf_no)

    else:
        base_dataset = torchvision.datasets.CIFAR100(root, train=True,
                                                     download=download)
        train_idxs, val_idxs = train_val_split(base_dataset.targets,
                                               num_classes,
                                               train=train)
        train_dataset = CIFAR100_train(root, train_idxs, args.percent,
                                       train=True, transforms=transforms)

        if train: # Validation sets
            val_dataset = CIFAR100_train(root, val_idxs, args.percent,
                                         train=True, transforms=[tf_no])
        else: # Test sets
            val_dataset = CIFAR100_val(root, val_idxs, train=False,
                                       transform=tf_no)


    if args.asym:
        train_dataset.asymmetric_noise()
        if train:
            val_dataset.asymmetric_noise()
    else:
        train_dataset.symmetric_noise()
        if train:
            val_dataset.symmetric_noise()

    if train:
        print(f"Has noisy targets? Train: {train_dataset.has_noisy_targets()}, Val: {val_dataset.has_noisy_targets()}")
    else:
        print(f"Has noisy targets? Train: {train_dataset.has_noisy_targets()}")

    return train_dataset, val_dataset


def train_val_split(train_val, num_classes, train=True, train_ratio=0.9):

    if train:
        train_val = np.array(train_val)

        train_n = int(len(train_val) * train_ratio / num_classes)
        train_idxs = []
        val_idxs = []

        for i in range(num_classes):
            idxs = np.where(train_val == i)[0]
            np.random.shuffle(idxs)
            train_idxs.extend(idxs[:train_n])
            val_idxs.extend(idxs[train_n:])
        np.random.shuffle(train_idxs)
        np.random.shuffle(val_idxs)
    else:
        train_idxs = None
        val_idxs = None

    return train_idxs, val_idxs


class CIFAR10_train(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, percent=0.0, train=True,
                 transforms=None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                                            transform=transforms,
                                            target_transform=target_transform,
                                            download=download)
        self.percent = percent
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets_noisy = np.copy(np.array(self.targets)[indexs])
            self.targets = np.array(self.targets)[indexs]
        else:
            self.targets_noisy = np.copy(np.array(self.targets))

        self.size = len(self.data)
        self.image_transforms = transforms

    def has_noisy_targets(self):
        return not np.array_equal(self.targets_noisy, self.targets)

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.data))
        for i, idx in enumerate(indices):
            if i < self.percent * len(self.data):
                self.targets_noisy[idx] = np.random.randint(10, dtype=np.int32)

    def asymmetric_noise(self):
        for i in range(10):
            indices = np.where(self.targets_noisy == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.percent * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.targets_noisy[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.targets_noisy[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.targets_noisy[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.targets_noisy[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.targets_noisy[idx] = 7

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        assert index < self.size
        img, target_noisy, target = \
            self.data[index], self.targets_noisy[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        transformed_imgs = list()
        for transform in self.image_transforms:
            images = img.copy()
            images = transform(images)
            transformed_imgs.append(images)

        if self.target_transform is not None:
            target_noisy = self.target_transform(target_noisy)

        return transformed_imgs, target_noisy, target


class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                                          transform=transform,
                                          target_transform=target_transform,
                                          download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]


class CIFAR100_train(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, percent=0.0, train=True,
                 transforms=None, target_transform=None,
                 download=False):
        super(CIFAR100_train, self).__init__(root, train=train,
                                            transform=transforms,
                                            target_transform=target_transform,
                                            download=download)
        self.percent = percent
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets_noisy = np.copy(np.array(self.targets)[indexs])
            self.targets = np.array(self.targets)[indexs]
        else:
            self.targets_noisy = np.copy(np.array(self.targets))

        self.size = len(self.data)
        self.image_transforms = transforms

    def has_noisy_targets(self):
        return not np.array_equal(self.targets_noisy, self.targets)

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.data))
        for i, idx in enumerate(indices):
            if i < self.percent * len(self.data):
                self.targets_noisy[idx] = np.random.randint(100, dtype=np.int32)

    def asymmetric_noise(self):
        with open('datasets/c100_supertosub.txt', 'r') as f: 
            content = f.read()
            superclass_to_subclasses = eval(content)

        with open('datasets/c100_subtosuper.txt', 'r') as f: 
            content = f.read()
            subclass_to_superclass = eval(content)

        indices = np.random.permutation(len(self.data))
        for i, idx in enumerate(indices):
            if i < self.percent * len(self.data):
                target = self.targets_noisy[idx]
                target_name = self.classes[target]
                superclass_name = subclass_to_superclass[target_name]
                subclasses = superclass_to_subclasses[superclass_name]

                # Old way
                #other_subclasses = np.setdiff1d(subclasses, target_name)
                #index = np.random.randint(0, 4)
                #target_new_name = other_subclasses[index]
                #target_new = self.class_to_idx[target_new_name]
                #self.targets_noisy[idx] = target_new

                # New way
                index = (subclasses.index(target_name) + 1) % 5
                target_new_name = subclasses[index]
                target_new = self.class_to_idx[target_new_name]
                self.targets_noisy[idx] = target_new
                """
                subclasses_targets = [self.class_to_idx[e] for e in subclasses]
                print("---")
                print("Superclass name:", superclass_name)
                print("Subclass names:", subclasses)
                print("Subclass targets:", subclasses_targets)
                print("True target name:", target_name, target)
                print("New target name:", target_new_name, target_new)
                print("---")
                """
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        assert index < self.size
        img, target_noisy, target = \
            self.data[index], self.targets_noisy[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        transformed_imgs = list()
        for transform in self.image_transforms:
            images = img.copy()
            images = transform(images)
            transformed_imgs.append(images)


        if self.target_transform is not None:
            target_noisy = self.target_transform(target_noisy)

        return transformed_imgs, target_noisy, target

class CIFAR100_val(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_val, self).__init__(root, train=train,
                                          transform=transform,
                                          target_transform=target_transform,
                                          download=download)

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

"""
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.utils.data as data


def imshow(img, name):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(name)


if __name__ == "__main__":
    dataset_dir = '/local_storage/datasets/cifar/'

    transform_strong_aug = transforms.Compose([
        transforms.ToTensor(),
    ])

    with open('c100_subtosuper.txt', 'r') as f: 
        content = f.read()
        subclass_to_superclass = eval(content)

    base_dataset = torchvision.datasets.CIFAR100(dataset_dir, train=True,
                                                    download=False)
    train_idxs, val_idxs = train_val_split(base_dataset.targets,
                                            100,
                                            train=True)
    train_dataset = CIFAR100_train(dataset_dir, train_idxs, 0.4,
                                    train=True, transforms=[transform_strong_aug])
    train_dataset.asymmetric_noise()
    
    dataloader = data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    dataiter = iter(dataloader)
    images, y_noisy, y = dataiter.next()
    
    for i in range(8):
        label = y[i].item()
        label_n = y_noisy[i].item()
        print(f'T: {label}', train_dataset.classes[label], f'N: {label_n}', train_dataset.classes[label_n], subclass_to_superclass[train_dataset.classes[label]])

    imshow(torchvision.utils.make_grid(images[0]),'test0.png')
"""