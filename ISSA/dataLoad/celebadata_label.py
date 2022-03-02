import gzip
import numpy as np
import os
import copy
import pickle
import torch
import random
import time
from skimage.transform import rotate
from skimage.morphology import dilation
from skimage.filters import threshold_otsu
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def load_celeba_sets(data_dir, args, num_sets=10, selected=False):
    if (not selected):
        train_images = torch.load(os.path.join(data_dir, 'celeba64.pt'))
        # print(train_images.shape)
        train_labels = torch.load(os.path.join(data_dir, 'celebalabels64.pt'))
        # print(train_labels.shape)
        train_images = train_images / torch.max(train_images)  # normalized image to [0,1]

        test_images = torch.load(os.path.join(data_dir, 'Test_celeba64.pt'))
        test_labels = torch.load(os.path.join(data_dir, 'Test_celebalabels64.pt'))
        test_images = test_images / torch.max(test_images)

        # torch.save(train_images[:40000], 'celeba64_small.pt')
        # torch.save(train_labels[:40000], 'celebalabels64_small.pt')
        # torch.save(test_images[:100], 'Test_celeba64_small.pt')
        # torch.save(test_labels[:100], 'Test_celebalabels64_small.pt')
        # train_images = torch.load('celeba64_small.pt')
        # train_labels = torch.load('celebalabels64_small.pt')
        # test_images = torch.load('Test_celeba64_small.pt')
        # test_labels = torch.load('Test_celebalabels64_small.pt')
    else:
        train_images = torch.load(os.path.join(data_dir, 'celeba64trainIS.pt'))
        train_labels = torch.load(os.path.join(data_dir, 'celebalabelstrainIS.pt'))
        test_images = torch.load(os.path.join(data_dir, 'celeba64testIS.pt'))
        test_labels = torch.load(os.path.join(data_dir, 'celebalabelstestIS.pt'))

    train_dataset = CelebaSetsDataset(train_labels, args, sample_size=args.sample_size, imgSize=args.imageSize,
                                      num_sets=num_sets)
    test_dataset = CelebaSetsDataset(test_labels, args, sample_size=args.sample_size, imgSize=args.imageSize,
                                     num_sets=num_sets)

    idx = torch.randperm(len(train_dataset))[:args.batchSize]
    # train_sample = torch.utils.data.Subset(train_dataset, idx)
    # train_sample = train_dataset[:args.batchSize]
    train_sample = train_dataset[idx][0]
    train_sample = train_sample.view(args.batchSize * args.sample_size)
    train_sample_label_idx = train_dataset[idx][1]
    train_sample_label = train_labels[train_sample_label_idx]
    train_sample_label = F.one_hot(train_sample_label, num_classes=10178).float().to(args.device)

    idx = torch.randperm(len(test_dataset))[:args.batchSize]
    # test_sample = torch.utils.data.Subset(test_dataset, idx)
    # test_sample = test_dataset[:args.batchSize]
    test_sample = test_dataset[idx][0]
    test_sample = test_sample.view(args.batchSize * args.sample_size)
    test_sample_label = test_dataset[idx][0]

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batchSize,
                                   shuffle=True, num_workers=0, drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=(args.batchSize - (args.batchSize % args.sample_size)),
                                  shuffle=False, num_workers=0, drop_last=False)

    return (train_dataset.num_classes, train_loader, test_loader, train_sample, train_sample_label, test_sample, test_sample_label, train_images, test_images, train_labels, test_labels)


def resample_celeba_sets(train_labels, args, num_sets=10):
    train_dataset = CelebaSetsDataset(train_labels, args, sample_size=args.sample_size, imgSize=args.imageSize,
                                      num_sets=num_sets)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batchSize,
                                   shuffle=True, num_workers=0, drop_last=True)
    return train_loader


def load_celeba_raw(data_dir, selected=False):
    if (not selected):
        train_images = torch.load(data_dir + 'celeba64.pt')
        train_labels = torch.load(data_dir + 'celebalabels64.pt')
        train_images = train_images / torch.max(train_images)  # normalized image to [0,1]

        test_images = torch.load(data_dir + 'Test_celeba64.pt')
        test_labels = torch.load(data_dir + 'Test_celebalabels64.pt')
        test_images = test_images / torch.max(test_images)
    else:
        train_images = torch.load(data_dir + 'celeba64trainIS.pt')
        train_labels = torch.load(data_dir + 'celebalabelstrainIS.pt')
        test_images = torch.load(data_dir + 'celeba64testIS.pt')
        test_labels = torch.load(data_dir + 'celebalabelstestIS.pt')

    return (train_images, train_labels, test_images, test_labels)


class CelebaSetsDataset(data.Dataset):
    def __init__(self, labels, args, sample_size=5, imgSize=32, augment=False, num_sets=-1):
        self.sample_size = sample_size
        self.imgSize = imgSize
        self.nc = args.nc
        self.args = args
        unique = torch.unique(labels)
        self.num_classes = unique.shape[0]

        sets, set_labels = self.make_sets(labels, num_sets=num_sets)
        sets = sets.reshape(-1, self.sample_size)

        # sets = torch.load('trainset.pt')
        # set_labels = torch.load('trainlabels.pt')
        # print(sets.shape, set_labels.shape)
        # torch.save(sets[:1000, :], 'trainset.pt')
        # torch.save(set_labels[:1000], 'trainlabels.pt')

        print("make sets made :")
        self.n = len(sets)
        self.data = {
            'inputs': sets,
            'targets': set_labels
        }

    def __getitem__(self, item):
        data = self.data['inputs'][item]
        labels = self.data['targets'][item]
        return data, labels

    def get_label(self, idx):
        label = self.data['targets'][idx]
        return label

    def __len__(self):
        return self.n

    def make_sets(self, labels, num_sets=-1):
        unique = torch.unique(labels)
        num_classes = unique.shape[0]
        print(f'num classes: {num_classes}, using {num_classes // self.args.label_div}')
        n = len(labels)

        image_sets = []
        set_labels = []
        prev_idx = 0

        for i in range(num_classes // self.args.label_div):
            cid = unique[i]
            ix = (labels == cid).nonzero(as_tuple=True)[0].tolist()
            num_instances = len(ix)
            if num_instances < self.sample_size or num_instances < 20:
                pass
            else:
                sampled_ix = []
                for z in range(num_sets):
                    sampled_ix = sampled_ix + random.sample(ix, self.sample_size)
                out_ix = sampled_ix
                image_sets.append(torch.tensor(out_ix))
                set_labels.append(torch.tensor([cid] * num_sets))

        x = torch.cat(image_sets, axis=0)
        y = torch.cat(set_labels, axis=0)

        return x, y


def evaluation_loader(args, labels, sample_size, top=True):
    eval_dataset = EvaluationDataset(labels, sample_size, top=top)
    eval_loader = data.DataLoader(dataset=eval_dataset,
                                  batch_size=(args.batchSize - (args.batchSize % args.sample_size)),
                                  shuffle=False, num_workers=0, drop_last=False)
    return eval_loader


class EvaluationDataset(data.Dataset):
    def __init__(self, labels, sample_size, top=True):
        self.sample_size = sample_size
        sets, set_labels = self.make_sets(labels, top=top)
        # sets = sets.reshape(-1, self.sample_size)
        # print('eval set ', sets.shape)
        self.n = len(sets)
        self.sets = sets
        self.labels = set_labels

    def __getitem__(self, item):
        data = self.sets[item]
        label = self.labels[item]
        return data, label

    def get_label(self, idx):
        label = self.labels[idx]
        return label

    def __len__(self):
        return self.n

    def make_sets(self, labels, top=True):
        unique = torch.unique(labels)
        num_classes = unique.shape[0]
        n = len(labels)

        image_sets = []
        set_labels = []
        prev_idx = 0

        for i in range(num_classes):
            cid = unique[i]
            ix = (labels == cid).nonzero(as_tuple=True)[0].tolist()
            num_instances = len(ix)
            if num_instances < self.sample_size * 2 or num_instances < 20:
                pass
            else:
                if (top):
                    out_ix = ix[0:self.sample_size]
                else:
                    out_ix = ix[self.sample_size:self.sample_size * 2]
                image_sets.append(torch.tensor(out_ix))
                set_labels.append(torch.tensor([cid]))

        x = torch.cat(image_sets, axis=0)
        y = torch.cat(set_labels, axis=0)

        return x, y



