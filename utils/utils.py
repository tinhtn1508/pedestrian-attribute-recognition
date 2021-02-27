import os
import sys
from PIL import Image
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle
import torch

def gen_A(num_classes, t, adj_file):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def getDataInfo(path):
    with open(path, 'rb+') as f:
        dataset_info = pickle.load(f)
    return dataset_info

def default_loader(path):
    return Image.open(path).convert('RGB')

class MultiLabelDataset(data.Dataset):
    def __init__(self, split, dataset_info, transform = None, loader = default_loader):
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        self.transform = transform
        self.loader = loader
        self.root_path = dataset_info.root
        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)
        self.img_idx = dataset_info.partition[split]

        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = self.loader(imgpath)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(gt_label)

    def __len__(self):
        return len(self.img_id)

def GetDataset(desciptionFile: str, imageTrainSize: tuple = (256, 128)):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=imageTrainSize),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        normalize
        ])

    data_info = getDataInfo(desciptionFile)
    train_dataset = MultiLabelDataset(split="train",
                    dataset_info = data_info, transform=transform_train)
    test_dataset = MultiLabelDataset(split="test",
                    dataset_info = data_info, transform=transform_train)
    return train_dataset, test_dataset, data_info.attr_size, data_info.attr_name
