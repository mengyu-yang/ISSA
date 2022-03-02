import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import transforms
import torchvision.transforms.functional as TF
from argparse import Namespace
from skimage.transform import rotate
import numpy as np
import os

from models import densenet, nets
from dataLoad import omnidata
from util import utils
from util.utils import mkdir, truncated_z_sample
import random

def rescaleImg(x):

    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32,32)), 
                transforms.ToTensor()])

    x_list = torch.zeros(x.size(0),32,32)
    for i in range(x.size(0)):
        x_list[i,:,:] = transform(x[i,:,:])
    x_list = 2 * x_list - 1

    return x_list

def repeat_context(context, sample_size, c_dim, repeat_size):
    context = context.view(-1,sample_size,c_dim)
    repeated = torch.repeat_interleave(context, repeat_size, dim=0)
    repeated = repeated.view(-1,c_dim)
    return repeated

def repeat_labels(y, sample_size, repeat_size):
    y = y.view(-1, sample_size)
    repeated = torch.repeat_interleave(y, repeat_size, dim=0)
    repeated = repeated.view(-1)
    return repeated

'''
only use part of the images in the set to form the context
k is the number of images used
'''
def only_k(k, x, sample_size, nc, imgSize):
    x = x.view(-1,sample_size,nc,imgSize,imgSize)
    if (k == 1):
        for i in range(x.size(0)):
            for j in range(1, x.size(1)):
                x[i,j] = x[i,0]
    elif (k == 2):
        for i in range(x.size(0)):
            for j in range(2, x.size(1)):
                if (j % 2 == 0):
                    x[i,j] = x[i,0]
                else:
                    x[i,j] = x[i,1]
    elif (k == 10):
        for i in range(x.size(0)):
            for j in range(5, x.size(1)):
                x[i,j] = x[i,j-5]
    return x


'''
generate fake images based on real images from this batch
only returns batch of fake images
'''
def gen_samples(x_batch, y_batch, sample_size, encoder, generator, anno_rate, c_dim, nz):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to28 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28,28)), 
        transforms.ToTensor()])

    fixed_noise = torch.randn(int(x_batch.shape[0]*(anno_rate)), nz, device=device)

    with torch.no_grad():
        x_batch = rescaleImg(x_batch.cpu()) #turns into 32 * 32
        N = x_batch.size(0)
        '''
        careful here, uses only k images for context, specifically for 10shot
        '''
        #x_batch = only_k(5, x_batch, sample_size, 1, 32)

        context = encoder(x_batch.to(device))
        if (context.size(0) < N):
            N = context.shape[0]
            context = context.unsqueeze(1).repeat(1, sample_size, 1)
            context = context.view(N*sample_size, -1)
        context = repeat_context(context, sample_size, c_dim, anno_rate)
        gen_labels = repeat_labels(y_batch, sample_size, anno_rate)

        gen_samples = generator(fixed_noise, context)
        gen_samples = utils.reconstruct_original(gen_samples)
        gen_x = gen_samples.view(-1,1,32,32) 

    gen_x = gen_x.detach().clone().to(device)
    gen_labels = gen_labels.detach().clone().to(device)
    return gen_x, gen_labels


'''
shuffle the dataset in the way where images from the same class stick together
'''
def shuffle_set(x, y, sample_size, imgSize=28):
    
    y_perm = y.clone()
    x_perm = x.clone()
    y_perm = y_perm.view(-1, sample_size)
    x_perm = x_perm.view(-1, sample_size, imgSize, imgSize)
    perm = torch.randperm(y_perm.shape[0])

    y_perm = y_perm[perm,:]
    y_perm = y_perm.view(-1)
    x_perm = x_perm[perm,:,:,:]
    x_perm = x_perm.view(-1, 1, imgSize, imgSize)

    return x_perm, y_perm

'''
test if the generated fake images can be correctly classifered
'''
def test_with_augment(train_ixs, x_test, y_test, batch_size_test, network, encoder, generator, sample_size, c_dim, nz, imgSize=28, nc=1, anno_rate=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.eval()
    total = 0
    correct = 0
    top5_correct = 0
    top10_correct = 0


    for i in range(0, len(train_ixs), batch_size_test):
        stop = min(batch_size_test, len(train_ixs[i:]))
        batch_ix = train_ixs[i:i+stop]
        batch_x = x_test[batch_ix].to(device)
        batch_y = y_test[batch_ix].to(device)
        batch_x = batch_x.view(-1,nc,imgSize,imgSize)
        batch_x, batch_y = gen_samples(batch_x, batch_y, sample_size, encoder, generator, anno_rate, c_dim, nz) #just feed the augmented images

        if (batch_x.size(0) != batch_y.size(0)):
            print (batch_x.shape)
            print (batch_y.shape)

        total = total + batch_x.size(0)
        output = network(batch_x)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_y.view_as(pred)).sum()

        #top5 accuracy
        pred = torch.topk(output, 5, dim=1, largest=True)[1]
        for k in range(batch_y.size(0)):
            y = batch_y[k]
            if (y in pred[k]):
                top5_correct += 1

        #top 10 accuracy
        pred = torch.topk(output, 10, dim=1, largest=True)[1]
        for k in range(batch_y.size(0)):
            y = batch_y[k]
            if (y in pred[k]):
                top10_correct += 1

        del batch_x
        del batch_y
        torch.cuda.empty_cache()

    accu = correct.item()
    accu = accu / total

    top5accu = top5_correct
    top5accu = top5accu / total

    top10accu = top10_correct
    top10accu = top10accu / total

    return accu, top5accu, top10accu





'''
train with real data
'''
def train(epoch, x_train, y_train, batch_size_train, network, optimizer, augments="none"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.train()
    ridx = np.random.permutation(len(x_train))
    x_train = x_train.clone()[ridx]
    y_train = y_train.clone()[ridx]
    for i in range(0, len(x_train), batch_size_train):
        stop = min(batch_size_train, len(x_train[i:]))
        batch_x = x_train[i:i+stop].to(device)
        batch_y = y_train[i:i+stop].to(device)
        batch_x = batch_x.view(-1,1,28,28)
        batch_x = rescaleImg(batch_x.cpu())
        batch_x = utils.reconstruct_original(batch_x).to(device)
        copy_x = batch_x.detach().clone().to(device)       #copy original samples
        copy_y = batch_y.detach().clone().to(device)       #copy original labels
        copy_x = copy_x.view(-1,32,32)
        batch_x = batch_x.view(-1,1,32,32)
        if (augments != "none"):
            aug_x = DiffAugment(batch_x, augments)  #augment image for training
        aug_x = aug_x.view(-1,32,32)
        batch_x = torch.cat((copy_x,  aug_x), 0)  #also use the real samples in standard augmentation training 
        batch_y = torch.cat((copy_y,  batch_y), 0)

        optimizer.zero_grad()
        batch_x = batch_x.view(-1,1,32,32)
        output = network(batch_x)
        pred = output.data.max(1, keepdim=True)[1]
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()



def test(x, y, batch_size_test, network):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.eval()
    test_correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size_test):
            stop = min(batch_size_test, len(x[i:]))
            batch_x = x[i:i+stop].to(device)
            batch_x = batch_x.view(-1,1,28,28)
            batch_x = rescaleImg(batch_x.cpu())
            batch_x = utils.reconstruct_original(batch_x)
            batch_y = y[i:i+stop].to(device)
            batch_x = batch_x.view(-1,1,32,32).to(device)
            output = network(batch_x)
            pred = output.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(batch_y.view_as(pred)).sum()
    accu = test_correct.item()
    accu = accu / len(x)
    return accu



def train_test_split(x, y, num_train, num_test, num_val):
    img_idx=[]
    for i in range(y.shape[0]-1):
        if (y[i] != y[i+1]):
            img_idx.append(i+1)
    x_train = x[0:num_train]
    y_train = y[0:num_train]
    x_test = x[num_train:num_train+num_test]
    y_test = y[num_train:num_train+num_test]
    x_val = x[num_train+num_test:num_train+num_test+num_val]
    y_val = y[num_train+num_test:num_train+num_test+num_val]


    for i in range(1, len(img_idx)):
        x_train = torch.cat((x_train,x[img_idx[i-1]:img_idx[i-1]+num_train]),dim=0)
        y_train = torch.cat((y_train,y[img_idx[i-1]:img_idx[i-1]+num_train]),dim=0)
        
        x_test = torch.cat((x_test,x[img_idx[i-1]+num_train:img_idx[i-1]+num_train+num_test]),dim=0)
        y_test = torch.cat((y_test,y[img_idx[i-1]+num_train:img_idx[i-1]+num_train+num_test]),dim=0)
        
        x_val = torch.cat((x_val,x[img_idx[i]-num_val:img_idx[i]]),dim=0)
        y_val = torch.cat((y_val,y[img_idx[i]-num_val:img_idx[i]]),dim=0)

    print ("training set is" + str(x_train.shape))
    print ("test set is " + str(x_test.shape))
    print ("val set is " + str(x_val.shape))
    return x_train, y_train, x_test, y_test, x_val, y_val



'''
num_train indicates the images that could be sampled
'''
def sample_sets(y, min_num=20, num_sets=5, sample_size=5):
    unique = torch.unique(y)
    num_classes = unique.shape[0]
    print ("there are " + str(num_classes) + " number of test classes")

    out_ixs = []
    class_counter = 0

    for i in range(num_classes):
        cid = unique[i]
        ix = (y == cid).nonzero(as_tuple=True)[0].tolist()

        if (len(ix) >= min_num):
            class_counter += 1
            '''
            now randomly sample the sets
            '''        
            sampled_ix = []
            for z in range(num_sets):
                sampled_ix = sampled_ix + random.sample(ix, sample_size)
            out_ixs.append(torch.tensor(sampled_ix))
        else:
            pass

    out_ixs = torch.cat(out_ixs,dim=0)
    return out_ixs


def train_classifier():
    batch_size_train = 50   
    batch_size_test = 50     
    args.nc = 1
    learning_rate = 0.001 
    beta_1 = 0.9
    beta_2 = 0.99
    num_init_features = 64
    growth_rate = 64
    block_config = [3,3,3,3]
    drop_rate = 0.5

    #one additional channel for real or fake label thus enter 2
    network = densenet.DenseNet(growth_rate, block_config, num_init_features, 4, drop_rate, num_classes, nc=1)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
    network.to(device)

    print (n_epochs)
    best_accu = 0
    for epoch in range(0, n_epochs):
        train(epoch, train_x, train_y, batch_size_train, network, optimizer, augments=augments)
        val_accu = test(val_x, val_y, batch_size_test, network)
        print (" at epoch " + str(epoch) + " test domain validation accuracy is " + str(val_accu))
        if (val_accu > best_accu):
            best_accu = val_accu
            print ("best accuracy so far")
            test_accu = test(test_x, test_y, batch_size_test, network)
            print ("test domain test accuracy is " + str(test_accu))
            print ("saving best model")
            torch.save(network.state_dict(), 'omni_test18_32.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/data/', help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--c_dim', type=int, default=1000, help='dimension of context vector c')
    parser.add_argument('--nz', type=int, default=1000, help='size of the latent z vector')
    parser.add_argument('--sample_size', type=int, default=1, help='size of each set')
    parser.add_argument('--num_train', type=int, default=1, help='number of training samples per class (k-shot)')
    parser.add_argument('--num_gen', type=int, default=100, help='number of generated example per class')
    parser.add_argument('--num_sets', type=int, default=10, help='number of sampled sets per class')
    parser.add_argument('--anno_rate', type=int, default=1, help='number of generated samples per real sample')
    parser.add_argument('--augments', type=str, default='none', help='translation|cutout')
    parser.add_argument('--model_dir', type=str, default='specify where to load the model from', help='directory to load ISSA from')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--c_scale', type=float, default=1)
    parser.add_argument('--g_z_scale', type=float, default=1)
    parser.add_argument('--seed', type=int, default=2020)



    parser.add_argument('--g_sn', type=int, default=1)
    parser.add_argument('--num_svs', type=int, default=1)
    parser.add_argument('--nc', type=int, default=1, help='number of channels in input image')
    parser.add_argument('--ngf', type=int, default=400)
    parser.add_argument('--ndf', type=int, default=400)

    args = parser.parse_args()

    #can all pass these in the args

    n_epochs = args.n_epochs
    num_train = args.num_train
    anno_rate = args.anno_rate
    augments = args.augments
    c_dim = args.c_dim
    nz = args.nz    
    sample_size = args.sample_size
    c_scale = args.c_scale
    model_dir = args.model_dir




    batch_size_train = 10 * sample_size     #this will always be a multiplier of sample size 
    batch_size_test = 10 * sample_size      #this will always be a multiplier of sample size 
    log_interval = 5000


    random_seed = args.seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    omniglot_dir = os.getcwd() + args.dataroot + "/omniglot"
    x_train, y_train, x_eval, y_eval, x_test, y_test = omnidata.load_raw_omniglot(omniglot_dir, args, classifier_split=True)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    y_test = y_test - torch.min(y_test) #shift the index to be 0-400

    all_x = x_test
    all_y = y_test

    num_test = 1
    num_val = 1

    train_x, train_y, test_x, test_y, val_x, val_y = train_test_split(all_x, all_y, num_train, num_test, num_val)
    num_classes = 211


    '''
    block of code to train the classifier starts
    '''
    #train_classifier()

    '''
    block of code to train the classifier ends
    '''
    min_num = num_train
    train_ixs = sample_sets(train_y, min_num=min_num, num_sets=args.num_sets, sample_size=args.sample_size)
    anno_rate = int(args.num_gen / args.num_sets)

    args.nc = 1
    learning_rate = 0.001 
    beta_1 = 0.9
    beta_2 = 0.99


    print ("model_dir")
    encoder = nets.Encoder(args)
    generator = nets.Generator(args)
    
    generator.to(device)
    encoder.to(device)

    generator.load_state_dict(torch.load(model_dir+'generator.pt'))
    encoder.load_state_dict(torch.load(model_dir+'encoder.pt'))

    encoder.eval()
    generator.eval()

    num_init_features = 64
    growth_rate = 64
    block_config = [3,3,3,3]
    drop_rate = 0.5

    #one additional channel for real or fake label thus enter 2
    network = densenet.DenseNet(growth_rate, block_config, num_init_features, 4, drop_rate, num_classes, nc=1)
    cfname = 'omni_test18_32.pth'
    if (os.path.isfile(cfname)):
        network.load_state_dict(torch.load(cfname))
        print ("loaded " + str(cfname))
    network.to(device)


    test_accu, top5, top10 = test_with_augment(train_ixs, train_x, train_y, batch_size_test, network, encoder, generator, sample_size, c_dim, nz, 
            imgSize=28, nc=args.nc, anno_rate=anno_rate)
    
    print (" top1 accuracy is " + str(test_accu))
    print (" top5 accuracy is " + str(top5))
    print (" top10 accuracy is " + str(top10))






