import gzip
import numpy as np
import os
import copy
import pickle
import torch
import random
from skimage.transform import rotate
from skimage.morphology import dilation
from skimage.filters import threshold_otsu
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm

'''
loading sets of data 
'''
def load_omniglot_sets(data_dir, args, num_sets=-1, classifier_split=True, small=False):
    imgsize = args.imageSize
    batch_size = args.batchSize
    sample_size = args.sample_size
    args.nc = 1

    #there is randomness coming from makesets
    np.random.seed(args.seed)


    x = np.load(os.path.join(data_dir, 'omniglot_data.npy'))
    x = x / np.max(x)  #normalization step

    if (classifier_split):
        if small:
            x_train, x_test, x_val = x[:600], x[1412:1623], x[1200:1412]
            train_labels = np.arange(600)
            test_labels = np.arange(1412, 1623, 1)
            eval_labels = np.arange(1200, 1412, 1)
        else:
            x_train, x_test, x_val = x[:1200], x[1412:1623], x[1200:1412]
            train_labels = np.arange(1200)
            test_labels = np.arange(1412,1623,1)
            eval_labels = np.arange(1200,1412,1)
    else:
        if small:
            x_train, x_test, x_val = x[:600], x[1200:1600], x[1600:]
            train_labels = np.arange(600)
            test_labels = np.arange(1200, 1600, 1)
            eval_labels = np.arange(1600, 1622, 1)
        else:
            x_train, x_test, x_val = x[:1200], x[1200:1600], x[1600:]
            train_labels = np.arange(1200)
            test_labels = np.arange(1200,1600,1)
            eval_labels = np.arange(1600,1622,1)
    train_images = x_train.reshape((-1,28,28))
    train_labels = np.repeat(train_labels, 20) 

    test_images = x_test.reshape((-1,28,28))
    test_labels = np.repeat(test_labels, 20)

    eval_images = x_val.reshape((-1,28,28))
    eval_labels = np.repeat(eval_labels, 20)


    train_dataset = OmniglotSetsDataset(train_images, train_labels, args, sample_size=sample_size, augment=args.data_aug, num_sets=num_sets)
    eval_dataset = OmniglotSetsDataset(eval_images, eval_labels, args, sample_size=sample_size, augment=False, num_sets=num_sets)
    test_dataset = OmniglotSetsDataset(test_images, test_labels, args, sample_size=sample_size, augment=False, num_sets=num_sets)

    train_sample = train_dataset[:args.batchSize]
    train_sample = torch.tensor(train_sample)
    train_sample = train_sample.view(batch_size*sample_size, 1, 28, 28)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0, drop_last=True)

    eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, drop_last=True)

    return (train_loader, eval_loader, test_loader, train_sample, eval_dataset, test_dataset)

'''
just load the images and labels
'''
def load_raw_omniglot(data_dir, args, classifier_split=False, small=False):

    x = np.load(os.path.join(data_dir, 'omniglot_data.npy'))
    x = x / np.max(x)
    if (classifier_split):
        if small:
            x_train, x_test, x_val = x[:600], x[1412:1622], x[1200:1412]
            train_labels = np.arange(600)
            test_labels = np.arange(1412, 1622, 1)
            eval_labels = np.arange(1200, 1412, 1)
        else:
            x_train, x_test, x_val = x[:1200], x[1412:1622], x[1200:1412]
            train_labels = np.arange(600)
            test_labels = np.arange(1412,1622,1)
            eval_labels = np.arange(1200,1412,1)
    else:
        if small:
            x_train, x_test, x_val = x[:600], x[1200:1600], x[1600:]
            train_labels = np.arange(600)
            test_labels = np.arange(1200, 1600, 1)
            eval_labels = np.arange(1600, 1623, 1)
        else:
            x_train, x_test, x_val = x[:1200], x[1200:1600], x[1600:]
            train_labels = np.arange(1200)
            test_labels = np.arange(1200,1600,1)
            eval_labels = np.arange(1600,1623,1)

    train_labels = np.repeat(train_labels, 20)
    test_labels = np.repeat(test_labels, 20)
    eval_labels = np.repeat(eval_labels, 20)

    train_images = x_train.reshape((-1,28,28)) 
    eval_images = x_val.reshape((-1,28,28))
    test_images = x_test.reshape((-1,28,28))
    return (train_images, train_labels,eval_images, eval_labels,test_images, test_labels)


def make_omni_sets(args, train_images, train_labels,eval_images, eval_labels,test_images, test_labels, num_sets=-1):
    imgsize = args.imageSize
    batch_size = args.batchSize
    sample_size = args.sample_size
    args.nc = 1

    train_dataset = OmniglotSetsDataset(train_images, train_labels, args, sample_size=sample_size, augment=args.data_aug, num_sets=num_sets)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0, drop_last=True)
    return train_loader


def evaluation_loader(args, images, labels, sample_size, top=True):
    eval_dataset = EvaluationDataset(images, labels, sample_size, imgSize=28, top=top)
    eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=args.batchSize,
                               shuffle=False, num_workers=0, drop_last=False)
    return eval_loader


class EvaluationDataset(data.Dataset):
    def __init__(self, images, labels, sample_size, imgSize=28, top=True):
        self.sample_size = sample_size      
        self.imgSize = imgSize      #original size of omniglot
        self.num_classes = np.max(labels) + 1 - np.min(labels)
        self.nc = 1
        sets, set_labels = self.make_sets(images, labels, top=top)
        self.n = len(sets)
        assert self.n == self.num_classes        #always have number of sets equal to the number of classes
        assert self.sample_size <= 20       #can't have set size larger than 20
        self.sets = sets
        self.labels = set_labels

    def __getitem__(self, item):
        data = self.sets[item]
        return data

    def get_label(self, idx):
        label = self.labels[idx]
        return label

    def __len__(self):
        return self.n
        
    def make_sets(self, images, labels, top=True):
        sets = np.zeros((self.num_classes,self.sample_size,self.nc,self.imgSize,self.imgSize))
        images = images.reshape((self.num_classes,20,self.nc,self.imgSize,self.imgSize))
        if (top):
            sets = images[:,0:self.sample_size,:,:,:]
        else:
            sets = images[:,20-self.sample_size:20,:,:,:]
        print(sets.shape)
        labels = labels.reshape((self.num_classes,20))
        set_labels = labels[:,0]
        set_labels = set_labels.reshape((self.num_classes))

        return sets, set_labels 




'''
return a dat
'''
def load_omniglot(args):
    data_dir = os.getcwd() + args.dataroot + '/{}'.format(args.dataset)
    imgsize = args.imageSize
    batch_size = args.batchSize
    if (args.dataset == "omniglot"):
        nc = 1
    else:
        raise NotImplementedError('Dataset not supported yet.')

    # create datasets
    path = os.path.join(data_dir, 'train_val_test_split.pkl')
    with open(path, 'rb') as file:
        splits = pickle.load(file)

    #transform to 32 * 32
    transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize(args.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])


    #train
    train_images, train_labels = splits[:2]
    eval_images, eval_labels = splits[2:4]
    if (args.data_aug):
        augment=True
    else:
        augment=False
    train_dataset = OmniglotDataset(train_images, train_labels, nc, imgsize, transform=transform, augment=augment)
    eval_dataset = OmniglotDataset(eval_images, eval_labels, nc, imgsize, transform=transform)

    #test
    test_images, test_labels = splits[4:]
    test_dataset = OmniglotDataset(test_images, test_labels, nc, imgsize, transform=transform)

    # create loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

    X_training = torch.zeros(len(train_loader), nc, imgsize, imgsize)
    Y_training = torch.zeros(len(train_loader))
    for i, x in enumerate(train_loader):
        X_training[i, :, :, :] = x[0]
        Y_training[i] = x[1]
        if i % 10000 == 0:
            print('Loading data... {}/{}'.format(i, len(train_loader)))

    X_test = torch.zeros(len(test_loader), nc, imgsize, imgsize)
    Y_test = torch.zeros(len(test_loader))
    for i, x in enumerate(test_loader):
        X_test[i, :, :, :] = x[0]
        Y_test[i] = x[1]
        if i % 1000 == 0:
            print('i: {}/{}'.format(i, len(test_loader)))

    Y_training = Y_training.type('torch.LongTensor')
    Y_test = Y_test.type('torch.LongTensor')

    dat = {'X_train': X_training, 'Y_train': Y_training, 'X_test': X_test, 'Y_test': Y_test, 'nc': nc}
    return (dat)






class OmniglotDataset(data.Dataset):
    def __init__(self, imgs, labels, nc, imgSize, transform=None, augment=False):
        imgs = np.reshape(imgs, (-1,28,28))
        self.imgSize = imgSize
        self.nc = nc
        self.transform = transform
        if (augment is True):
            imgs, labels = self.augment_img(imgs, labels)
        imgs = torch.from_numpy(imgs)
        labels = torch.from_numpy(labels)
        self.imgs = np.reshape(imgs, (-1,28,28))
        self.labels = labels
        assert len(self.imgs) == len(self.labels)


    def augment_img(self, imgs, labels):
        out = imgs
        out_label = labels
        max_label = np.max(labels) + 1

        #1.horizontal flip
        augmented = np.copy(imgs)
        aug_labels = np.copy(labels)
        n_len = len(augmented)
        for s in range(n_len):
            augmented[s] = augmented[s, :, ::-1]
            aug_labels[s] = aug_labels[s] + max_label
        out = np.concatenate([out, augmented])
        out_label = np.concatenate([out_label, aug_labels])

        #2.vertical flip
        augmented = np.copy(imgs)
        aug_labels = np.copy(labels)
        for s in range(n_len):
            augmented[s] = augmented[s, ::-1, :]
            aug_labels[s] = aug_labels[s] + max_label * 2
        out = np.concatenate([out, augmented])
        out_label = np.concatenate([out_label, aug_labels])

        ##3. rotation 90
        angle = 90
        augmented = np.copy(imgs)
        aug_labels = np.copy(labels)
        for s in range(n_len):
            augmented[s] = rotate(augmented[s], angle)
            aug_labels[s] = aug_labels[s] + max_label * 3
        out = np.concatenate([out, augmented])
        out_label = np.concatenate([out_label, aug_labels])

        #4 rotation 180
        angle = 180
        augmented = np.copy(imgs)
        aug_labels = np.copy(labels)
        for s in range(n_len):
            augmented[s] = rotate(augmented[s], angle)
            aug_labels[s] = aug_labels[s] + max_label * 4
        out = np.concatenate([out, augmented])
        out_label = np.concatenate([out_label, aug_labels])


        #5 rotation 270
        angle = 270
        augmented = np.copy(imgs)
        aug_labels = np.copy(labels)
        for s in range(n_len):
            augmented[s] = rotate(augmented[s], angle)
            aug_labels[s] = aug_labels[s] + max_label * 5
        out = np.concatenate([out, augmented])
        out_label = np.concatenate([out_label, aug_labels])
        return (out,out_label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.transform is not None:
            outimgs = self.transform(self.imgs[idx])
        else:
            outimgs = self.imgs[idx]
        return outimgs, self.labels[idx]



class OmniglotSetsDataset(data.Dataset):
    def __init__(self, images, labels, args, sample_size=5, imgSize=28, augment=False, transform=None, num_sets=-1):
        self.sample_size = sample_size      
        self.imgSize = 28      #original size of omniglot
        self.transform = transform
        self.dilate = args.dilate
        self.binary = args.binary

        if (self.transform is not None):
            images = torch.from_numpy(images)
            images = images.view(-1,28,28)
            images = self.transform(images)
            images = images.view(-1,1,self.imgSize,self.imgSize)
        sets, set_labels = self.make_sets(images, labels, num_sets=num_sets)
        if augment:
            sets = self.augment_sets(sets, dilate=self.dilate, binary=self.binary)
        sets = sets.reshape(-1, self.sample_size, 1, self.imgSize, self.imgSize)

        self.n = len(sets)
        self.data = {
            'inputs': sets,
            'targets': set_labels
        }

    def __getitem__(self, item):
        data = self.data['inputs'][item]
        return data

    def get_label(self, idx):
        label = self.data['targets'][idx]
        return label

    def __len__(self):
        return self.n

    '''
    1. original image
    2. binarization
    3. randomly apply dilation
    4. horizontal reflection & vertical reflection
    5. 90, 180, 270 rotation
    '''
    def augment_sets(self, sets, dilate=True, binary=True):
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size, self.imgSize, self.imgSize)
        n_sets = len(augmented)
        total = sets

        #1.binarization
        if (binary):
            for s in range(n_sets):
                img = augmented[s]
                img = img.reshape((-1,28,28))
                threshold = threshold_otsu(img)
                img = img > threshold
                if (dilate):
                    for i in range(self.sample_size):
                        if (np.random.choice([0, 1])):
                            img[i] = dilation(img[i])
                img = img.reshape((-1, self.sample_size, self.imgSize, self.imgSize))
                augmented[s] = img

            total = np.concatenate([total, augmented.reshape(n_sets, self.sample_size, self.imgSize*self.imgSize)])
            augmented = np.copy(sets)
            augmented = augmented.reshape(-1, self.sample_size, self.imgSize, self.imgSize)

        #2. horizontal reflection 
        for s in range(n_sets):
            augmented[s] = augmented[s, :, :, ::-1]
            if (dilate):
                for item in range(self.sample_size):
                    if (np.random.choice([0, 1])):
                        augmented[s, item] = dilation(augmented[s, item])

        total = np.concatenate([total, augmented.reshape(n_sets, self.sample_size, self.imgSize*self.imgSize)])
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size, self.imgSize, self.imgSize)


        #3. vertical reflection
        for s in range(n_sets):
            augmented[s] = augmented[s, :, ::-1, :]
            if (dilate):
                for item in range(self.sample_size):
                    if (np.random.choice([0, 1])):
                        augmented[s, item] = dilation(augmented[s, item])

        total = np.concatenate([total, augmented.reshape(n_sets, self.sample_size, self.imgSize*self.imgSize)])
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size, self.imgSize, self.imgSize)

        #4. rotation 90 degree
        angle = 90
        for s in range(n_sets):
            for item in range(self.sample_size):
                augmented[s, item] = rotate(augmented[s, item], angle)
            if (dilate):
                for item in range(self.sample_size):
                    if (np.random.choice([0, 1])):
                        augmented[s, item] = dilation(augmented[s, item])
        total = np.concatenate([total, augmented.reshape(n_sets, self.sample_size, self.imgSize*self.imgSize)])
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size, self.imgSize, self.imgSize)


        #5. rotation 270 degree
        angle = 270
        for s in range(n_sets):
            for item in range(self.sample_size):
                augmented[s, item] = rotate(augmented[s, item], angle)
            if (dilate):
                for item in range(self.sample_size):
                    if (np.random.choice([0, 1])):
                        augmented[s, item] = dilation(augmented[s, item])
        total = np.concatenate([total, augmented.reshape(n_sets, self.sample_size, self.imgSize*self.imgSize)])
        return total



    @staticmethod
    def one_hot(dense_labels, num_classes):
        num_labels = len(dense_labels)
        offset = np.arange(num_labels) * num_classes
        one_hot_labels = np.zeros((num_labels, num_classes))
        one_hot_labels.flat[offset + dense_labels.ravel()] = 1
        return one_hot_labels

    '''
    num_sets is the number of sets per image class, it can be go higher by resampling from the same images
    -1 means that it is decided based on the number of images available with each class and each image is only included in one set 
    '''
    def make_sets(self, images, labels, num_sets=-1):
        num_classes = np.max(labels) + 1
        labels = self.one_hot(labels, num_classes)

        n = len(images)
        perm = np.random.permutation(n)
        images = images[perm]
        labels = labels[perm]

        image_sets = []
        set_labels = []

        for i in range(num_classes):
            ix = labels[:, i].astype(int)
            num_instances_of_class = np.sum(ix)
            ix = np.where(ix!=0)
            ix = ix[0]
            ix = ix.tolist()
            if num_instances_of_class < self.sample_size:
                pass
            elif (num_sets==-1):
                remainder = num_instances_of_class % self.sample_size
                image_set = images[ix]
                if remainder > 0:
                    image_set = image_set[:-remainder]
                image_sets.append(image_set)
                k = len(image_set)
                set_labels.append(labels[ix][:int(k / self.sample_size)])
            else:
                sampled_ix = []
                for z in range(num_sets):
                    sampled_ix = sampled_ix + random.sample(ix, self.sample_size)
                out_ix = sampled_ix
                image_set = images[out_ix]
                image_sets.append(image_set)
                set_labels.append(labels[out_ix][:int(len(image_set) / self.sample_size)])

        x = np.concatenate(image_sets, axis=0).reshape(-1, self.sample_size, self.imgSize*self.imgSize)
        y = np.concatenate(set_labels, axis=0)
        if np.max(x) > 1:
            x /= 255

        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]

        return x, y