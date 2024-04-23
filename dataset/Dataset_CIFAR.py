import logging
import math
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar10_test_dataset(Dataset): 
    def __init__(self, transform, root_dir):
        self.transform = transform
                       
        test_dic = unpickle('%s/test_batch'%root_dir)
        self.test_data = test_dic['data']
        self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))  
        self.test_label = test_dic['labels']
        self.test_label = np.array(self.test_label)
        self.test_label -= 2
        self.test_label[np.where(self.test_label == -2)[0]] = 8
        self.test_label[np.where(self.test_label == -1)[0]] = 9
        ood = np.where(self.test_label>5)
        self.test_label[ood] = 6

    def __getitem__(self, index):                         
        img, target = self.test_data[index], self.test_label[index]
        img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.test_label)
        
def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    # change label of CIFAR10
    # Animals 2 - 7
    # change to 0 - 5
    # labeled 0 - 5
    # unlabeled 6 - 9
    base_dataset.targets = np.array(base_dataset.targets)
    base_dataset.targets -= 2
    base_dataset.targets[np.where(base_dataset.targets == -2)[0]] = 8
    base_dataset.targets[np.where(base_dataset.targets == -1)[0]] = 9
    
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)
    train_labeled_dataset.targets = np.array(train_labeled_dataset.targets)
    train_labeled_dataset.targets -= 2
    
    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),target_transform='unlabel')
    
    test_dataset = cifar10_test_dataset(transform=transform_val, root_dir = './data/cifar-10-batches-py')

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def x_u_split(args, labels):
    # labeled 0-5 unlabeled 6-9
    labeled_class = args.known_class        
    label_per_class = args.label_per_class
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(labeled_class):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform=='unlabel':
            return img
        else:
            return img, target

# transform label of CIFAR100
# change known class to 0 - args.known_class 
def label_transform(args, coarse_labels):
    known_class = args.known_class
    label_trans_dict = {}
    sum_label = 0
    for i in range(len(coarse_labels)):
        if coarse_labels[i]<known_class/5:
            label_trans_dict[i] = sum_label
            sum_label+=1
        else:
            label_trans_dict[i] = 100
    return label_trans_dict

def change_label(label, label_trans_dict):
    for i in range(len(label)):
        label[i] = label_trans_dict[label[i]]
    return label

class cifar100_test_dataset(Dataset): 
    def __init__(self, args, transform, root_dir):
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13]) 
        self.transform = transform        
        test_dic = unpickle('%s/test'%root_dir)
        self.test_data = test_dic['data']
        self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))  
        self.test_label = test_dic['fine_labels'] 
        label_trans_dict = label_transform(args, coarse_labels)
        self.test_label = change_label(self.test_label, label_trans_dict)

    def __getitem__(self, index):                         
        img, target = self.test_data[index], self.test_label[index]
        img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target

    def __len__(self):
        return len(self.test_label)
        
def get_cifar100(args, root):
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13]) 
    
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    base_dataset = datasets.CIFAR100(root, train=True, download=True)
    label_trans_dict = label_transform(args, coarse_labels)
    base_dataset.targets = change_label(base_dataset.targets, label_trans_dict)    
    
    train_labeled_idxs, train_unlabeled_idxs = x_u_split_cifar100(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args, root, train_labeled_idxs, train=True,
        transform=transform_labeled)
    train_labeled_dataset.targets = change_label(train_labeled_dataset.targets, label_trans_dict) 
    
    train_unlabeled_dataset = CIFAR100SSL(
        args, root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std),target_transform='unlabel')
    
    test_dataset = cifar100_test_dataset(args, transform=transform_val, root_dir = './data/cifar-100-python')

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def x_u_split_cifar100(args, labels):
    known_class = args.known_class
    label_per_class = args.label_per_class
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(known_class):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)    
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, args, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.known_class = args.known_class
        self.coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13]) 

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform=='unlabel':
            return img
        else:
            return img, target

DATASET_GETTERS = {'cifar100': get_cifar100,
             'cifar10': get_cifar10}    