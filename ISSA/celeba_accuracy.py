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

from models import nets
from dataLoad import celebadata
'''
install your own pre-trained insight face classifier from https://github.com/deepinsight/insightface
and define it in insightclassifier
requires two function, 
get_model : returns a reference to the classifier as network
test_sample : returns the top1, top5 and top10 test accuracy
see the code for the use cases
'''
from insightclassifier import get_model, test_sample
from util import utils
from util.utils import mkdir, truncated_z_sample
import random

'''
rescale to -1,1 range for tanh
'''
def rescaleImg(x):
    x = 2 * x - 1
    return x

def repeat_context(context, c_dim, repeat_size):
    context = context.view(-1, c_dim)
    repeated = torch.repeat_interleave(context, repeat_size, dim=0)
    repeated = repeated.view(-1,c_dim)
    return repeated

def repeat_labels(y, sample_size, repeat_size):
    y = y.view(-1, sample_size)
    y = y[:,0]
    y = y.view(-1)
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
anno_rate = number of samples per set
'''
def gen_samples(x_batch, y_batch, sample_size, encoder, generator, anno_rate, c_dim, nz, imgSize=64, nc=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_noise = torch.randn(int(x_batch.shape[0] / sample_size *(anno_rate)), nz, device=device)

    with torch.no_grad():
        x_batch = rescaleImg(x_batch) #rescales to [-1,1]
        N = x_batch.size(0)
        context = encoder(x_batch)
        context = repeat_context(context, c_dim, anno_rate)
        gen_labels = repeat_labels(y_batch, sample_size, anno_rate)
        gen_samples = generator(fixed_noise, context)
        gen_samples = utils.reconstruct_original(gen_samples)
        gen_samples = gen_samples.view(-1,nc,imgSize,imgSize)
    gen_samples = gen_samples.to(device)
    gen_labels = gen_labels.to(device)
    return gen_samples, gen_labels

'''
shuffle the dataset in the way where images from the same class stick together
'''
def shuffle_set(x, sample_size):
    
    x_perm = x.clone()
    x_perm = x_perm.view(-1, sample_size)
    perm = torch.randperm(x_perm.shape[0])

    x_perm = x_perm[perm,:]
    x_perm = x_perm.view(-1)

    return x_perm


'''
test if the generated fake images can be correctly classifered
'''
def test_with_augment(batch_ixs, x_test, y_test, batch_size_test, network, encoder, generator, sample_size, c_dim, nz, imgSize=64, nc=3, anno_rate=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0
    correct = 0
    top5_correct = 0
    top10_correct = 0


    for i in range(0, len(batch_ixs), batch_size_test):
        stop = min(batch_size_test, len(batch_ixs[i:]))
        batch_ix = batch_ixs[i:i+stop]
        batch_x = x_test[batch_ix].to(device)

        '''
        careful here
        '''
        #batch_x = only_k(5, batch_x, sample_size, nc, imgSize)

        batch_x = batch_x.view(-1,nc,imgSize,imgSize)
        batch_y = y_test[batch_ix].to(device)
        batch_x, batch_y = gen_samples(batch_x, batch_y, sample_size, encoder, generator, anno_rate, c_dim, nz, imgSize=imgSize, nc=nc) #just feed the augmented images

        if (batch_x.size(0) != batch_y.size(0)):
            print (batch_x.shape)
            print (batch_y.shape)

        total = total + batch_x.size(0)
        c1, c5, c10 = test_sample(batch_x, batch_y, network)

        correct = correct + c1
        top5_correct = top5_correct + c5
        top10_correct = top10_correct + c10


    accu = correct.item()
    accu = accu / total

    top5accu = top5_correct
    top5accu = top5accu / total

    top10accu = top10_correct
    top10accu = top10accu / total

    return accu, top5accu, top10accu




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



'''
classes with number of images <= 2*num_val will not have any validation sample
'''
def train_val_split(x, y, num_train, ids=[]):

    train_ixs = []
    val_ixs = []
    nclasses = 0
    train_labels = []
    test_labels = []

    for i in range(len(ids)):
        cid = ids[i]
        nclasses += 1
        ix = (y == cid).nonzero(as_tuple=True)[0]
        train_len = len(ix[0:num_train])
        test_len = len(ix[num_train:])
        train_ixs.append(ix[0:num_train])
        val_ixs.append(ix[num_train:])
        train_labels.append(torch.tensor([i]*train_len))
        test_labels.append(torch.tensor([i]*test_len))
        
    print ("using " + str(nclasses) + " classes")
    train_ixs = torch.cat(train_ixs,dim=0)
    val_ixs = torch.cat(val_ixs,dim=0)
    train_labels = torch.cat(train_labels,dim=0)
    test_labels = torch.cat(test_labels,dim=0)

    return train_ixs, val_ixs, train_labels, test_labels


def get_class_distribution(labels):
    unique = torch.unique(labels)
    num_classes = unique.shape[0]
    class_counts = []
    class_ids = []
    for i in range(num_classes):
        cid = unique[i]
        ix = (labels == cid).nonzero(as_tuple=True)[0].tolist()
        class_counts.append(len(ix))
        class_ids.append(cid)
    return class_counts, class_ids




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='data/celeba/', help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--c_dim', type=int, default=100, help='dimension of context vector c')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--sample_size', type=int, default=1, help='size of each set')
    parser.add_argument('--num_gen', type=int, default=100, help='number of generated example per class')
    parser.add_argument('--num_sets', type=int, default=10, help='number of sampled sets per class')
    parser.add_argument('--model_dir', type=str, default='specify where the model is', help='directory to load ISSA from')
    parser.add_argument('--c_scale', type=float, default=1)
    parser.add_argument('--g_z_scale', type=float, default=1)
    parser.add_argument('--g_sn', type=int, default=1)
    parser.add_argument('--num_svs', type=int, default=1)
    parser.add_argument('--nc', type=int, default=3, help='number of channels in input image')
    parser.add_argument('--ngf', type=int, default=512)
    parser.add_argument('--ndf', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2020)


    args = parser.parse_args()

    #can all pass these in the args

    c_dim = args.c_dim
    nz = args.nz    
    sample_size = args.sample_size
    c_scale = args.c_scale
    model_dir = args.model_dir

    batch_size_test = sample_size * 64      #this will always be a multiplier of sample size 
    log_interval = 5000
    random_seed = args.seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    topk = 200
    train_x, train_y, test_x, test_y = celebadata.load_celeba_raw(args.dataroot, selected=False)
    all_y = test_y
    all_x = test_x
    test_counts, test_ids = get_class_distribution(all_y)
    test_counts = torch.tensor(test_counts)
    test_ids = torch.tensor(test_ids)
    test_sort = torch.argsort(test_counts, descending=True)
    test_counts = test_counts[test_sort]
    test_ids = test_ids[test_sort]
    test_ids = test_ids[0:topk]

    num_train = 29
    print ("using " + str(num_train) + " images per class as training for classifier")
    train_ixs, test_ixs, train_y, test_y = train_val_split(all_x, all_y, num_train, ids=test_ids.tolist())
    train_x = all_x[train_ixs]
    test_x = all_x[test_ixs]

    min_num = 20
    train_ixs = sample_sets(train_y, min_num=min_num, num_sets=args.num_sets, sample_size=args.sample_size)
    anno_rate = int(args.num_gen / args.num_sets)

    args.nc = 3
    
    if (anno_rate > 0):
        encoder = nets.Encoder(args)
        generator = nets.Generator(args)
        
        generator.to(device)
        encoder.to(device)

        encoder.eval()
        generator.eval()

        generator.load_state_dict(torch.load(model_dir+'generator.pt'))
        encoder.load_state_dict(torch.load(model_dir+'encoder.pt'))
    else:
        encoder = None
        generator = None

    print (train_x.shape)
    print (train_y.shape)



    network = get_model(train_x, train_y, topk=topk)        

    test_accu, top5, top10 = test_with_augment(train_ixs, train_x, train_y, batch_size_test, network, encoder, generator, sample_size, c_dim, nz, 
            imgSize=64, nc=3, anno_rate=anno_rate)
    
    print (" top1 accuracy is " + str(test_accu))
    print (" top5 accuracy is " + str(top5))
    print (" top10 accuracy is " + str(top10))


