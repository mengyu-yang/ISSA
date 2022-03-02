import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import os

from dataLoad import omnidata



def rescaleImg(x):
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32,32)), 
                transforms.ToTensor()])

    x_list = torch.zeros(x.size(0),32,32)
    for i in range(x.size(0)):
        x_list[i,:,:] = transform(x[i,:,:])

    return x_list


class Net(nn.Module):
    def __init__(self, num_classes=1623):
        super(Net, self).__init__()
        self.imgSize = 32
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.last = nn.Sequential(
            nn.Linear(4096, 2000, bias=False),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Linear(2000, num_classes, bias=False),
            nn.BatchNorm1d(num_classes)
        )

    # #this is the forward function to be used as a classifier
    # def forward(self, x):
    #     N = x.size(0)
    #     # x = x.view(N,-1)
    #     x = self.main(x)
    #     x = x.view(N,-1)
    #     x = self.last(x)
    #     return (F.log_softmax(x, dim=1), x)

    #this is the forward function to be used as feature extractor
    def features(self, x):
        N = x.size(0)
        # x = x.view(N,-1)
        x = self.main(x)
        return x

    #this is the forward function to be used as feature extractor
    def forward(self, x):
        N = x.size(0)
        # x = x.view(N,-1)
        x = self.main(x)
        return x


def train(epoch, x_train, y_train, batch_size_train, network):
    network.train()
    correct = 0
    ridx = np.random.permutation(len(x_train))
    x_train = x_train.clone()[ridx]
    y_train = y_train.clone()[ridx]
    for i in range(0, len(x_train), batch_size_train):
        stop = min(batch_size_train, len(x_train[i:]))
        batch_x = x_train[i:i+stop]
        batch_x = rescaleImg(batch_x).to(device)
        batch_x = batch_x.view(-1,1,32,32)
        batch_y = y_train[i:i+stop].to(device)
        optimizer.zero_grad()
        output, linear = network(batch_x)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_y.view_as(pred)).sum()
        loss = F.nll_loss(output, batch_y)
        loss.backward()
        optimizer.step()

    print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    correct, len(x_train),
    100. * correct / len(x_train)))


def train_val_split(x, y, num_val):
    img_idx=[]

    for i in range(y.shape[0]-1):
        if (y[i] != y[i+1]):
            img_idx.append(i+1)
    x_train = x[0:img_idx[0]-num_val]
    y_train = y[0:img_idx[0]-num_val]
    x_val = x[img_idx[0]-num_val:img_idx[0]]
    y_val = y[img_idx[0]-num_val:img_idx[0]]

    for i in range(1, len(img_idx)):
        train_num = img_idx[i]-num_val-img_idx[i-1]
        test_num = num_val 
        if (train_num < test_num):
            print("class " + str(i) + "has less training images than test")
        x_train = torch.cat((x_train,x[img_idx[i-1]:img_idx[i]-num_val]),dim=0)
        y_train = torch.cat((y_train,y[img_idx[i-1]:img_idx[i]-num_val]),dim=0)
        x_val = torch.cat((x_val,x[img_idx[i]-num_val:img_idx[i]]),dim=0)
        y_val = torch.cat((y_val,y[img_idx[i]-num_val:img_idx[i]]),dim=0)

    return x_train, y_train, x_val, y_val



def val(x_val, y_val, batch_size_test, network, prev_accu):
    print ("starting validating")
    network.eval()
    test_correct = 0
    with torch.no_grad():
        for i in range(0, len(x_val), batch_size_test):
            stop = min(batch_size_test, len(x_val[i:]))
            batch_x = x_val[i:i+stop]
            batch_x = rescaleImg(batch_x).to(device)
            batch_x = batch_x.view(-1,1,32,32)
            batch_y = y_val[i:i+stop].to(device)
            output, linear = network(batch_x)
            pred = output.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(batch_y.view_as(pred)).sum()
        print('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_correct, len(x_val),
            100. * test_correct / len(x_val)))

    accu = test_correct
    if (prev_accu < accu):
        print ("saving model")
        torch.save(network.state_dict(), 'omni_all_newsplit.pth')
    return accu






def train_test_split(x, y):
    y, indices = torch.sort(y)
    x = x[indices]
    img_idx=[]

    for i in range(y.shape[0]-1):
        if (y[i] != y[i+1]):
            img_idx.append(i+1)
    x_train = x[0:img_idx[0]-5]
    y_train = y[0:img_idx[0]-5]
    x_test = x[img_idx[0]-5:img_idx[0]]
    y_test = y[img_idx[0]-5:img_idx[0]]

    for i in range(1, len(img_idx)):
        train_num = img_idx[i]-5 - img_idx[i-1]
        test_num = 5 
        if (train_num < test_num):
            print("class " + str(i) + "has less training images than test")
        x_train = torch.cat((x_train,x[img_idx[i-1]:img_idx[i]-5]),dim=0)
        y_train = torch.cat((y_train,y[img_idx[i-1]:img_idx[i]-5]),dim=0)
        x_test = torch.cat((x_test,x[img_idx[i]-5:img_idx[i]]),dim=0)
        y_test = torch.cat((y_test,y[img_idx[i]-5:img_idx[i]]),dim=0)

    return x_train, y_train, x_test, y_test

            




if __name__ == '__main__':

    #python -u omni_classifier_BN.py

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='omniglot', help='omniglot')
    parser.add_argument('--dataroot', type=str, default='/data/', help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--n_classes', type=int, default=1200, help='number of classes used for conditioning')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--data_aug', type=int, default=0, help='augment the training dataset with flipping and rotation or not')

    args = parser.parse_args()

    n_epochs = 200
    batch_size_train = 128
    batch_size_test = 1000      
    log_interval = 5000


    random_seed = 2020
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    omniglot_dir = os.getcwd() + args.dataroot + "/omniglot"
    x_train, y_train, x_eval, y_eval, x_test, y_test = omnidata.load_raw_omniglot(omniglot_dir, args, classifier_split=True)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_eval = torch.from_numpy(x_eval)
    y_eval = torch.from_numpy(y_eval)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    all_x = torch.cat((x_train, x_eval, x_test), 0)
    all_y = torch.cat((y_train, y_eval, y_test), 0)
    num_val = 2 #two validation samples per class
    train_x, train_y, val_x, val_y = train_val_split(all_x, all_y, num_val)
    num_classes = 1623

    args.nc = 1

    learning_rate = 0.001 #follow closer look few shot paper
    network = Net(num_classes=num_classes)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    print (network)
    network.to(device)


    print (n_epochs)
    best_accu = 0
    for epoch in range(0, n_epochs):
        train(epoch, train_x, train_y, batch_size_train, network)
        accu = val(val_x, val_y, batch_size_test, network, best_accu)
        if (accu > best_accu):
            best_accu = accu

