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
import random
import itertools


from models import nets
from dataLoad import celebadata
from util import utils
from util.utils import mkdir, truncated_z_sample
from fid_score import calculate_fid_celeba, get_model

'''
rescale to -1,1 range for tanh
'''
def rescaleImg(x):
    x = 2 * x - 1
    return x

def repeat_context(c, sample_size):
    N = c.shape[0]
    c = c.unsqueeze(1).repeat(1, sample_size, 1)
    c = c.view(N*sample_size, -1)
    return c

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
    parser.add_argument('--dataroot', type=str, default='data/celeba', help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--c_dim', type=int, default=100, help='dimension of context vector c')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--sample_size', type=int, default=1, help='size of each set')
    parser.add_argument('--num_support', type=int, default=1, help='size of the actual support set')
    parser.add_argument('--num_gen', type=int, default=100, help='number of generated example per class')
    parser.add_argument('--num_sets', type=int, default=10, help='number of sampled sets per class')
    parser.add_argument('--model_dir', type=str, default='specify where to load model', help='specify where to load ISSA')
    parser.add_argument('--output_dir', required=True, help='')
    parser.add_argument('--c_scale', type=float, default=1)
    parser.add_argument('--g_z_scale', type=float, default=1)
    parser.add_argument('--g_sn', type=int, default=1)
    parser.add_argument('--num_svs', type=int, default=1)
    parser.add_argument('--nc', type=int, default=3, help='number of channels in input image')
    parser.add_argument('--ngf', type=int, default=512)
    parser.add_argument('--ndf', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2020)



    args = parser.parse_args()
    mkdir(os.path.join(args.output_dir, 'viz_sample'))
    args.output_dir = os.getcwd() + args.output_dir
    print (args.output_dir)


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
    args.nc = 3

    print (args)

    encoder = nets.Encoder(args)
    generator = nets.Generator(args)
    
    generator.to(device)
    encoder.to(device)

    encoder.eval()
    generator.eval()

    generator.load_state_dict(torch.load(model_dir+'generator.pt'))
    encoder.load_state_dict(torch.load(model_dir+'encoder.pt'))


    args.batchSize = 200
    celeba_dir = 'data/celeba'
    args.selected=0
    train_loader, test_loader, train_sample, test_sample, train_images, test_images, train_labels, test_labels = celebadata.load_celeba_sets(celeba_dir, args, 
                num_sets=args.num_sets, selected=args.selected)

    fid_model = get_model(device, dataset="celeba")
    '''
    use 10 images from each class as evaluation always
    '''
    eval_size = 10
    context_test_set = celebadata.evaluation_loader(args, test_labels, eval_size, top=True)
    all_test_set = celebadata.evaluation_loader(args, test_labels, eval_size, top=False)

    dims = 2048
    imgPclass = 10
    test_fid = calculate_fid_celeba(context_test_set, all_test_set, test_images, encoder, generator, device, args, 
                        dims=dims, model=fid_model, imgPclass=imgPclass, eval_size=eval_size, num_support=args.num_support)
    print ("test fid is " + str(test_fid))

        

