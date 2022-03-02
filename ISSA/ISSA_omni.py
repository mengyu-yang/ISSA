from __future__ import print_function

#external
import argparse
import itertools
import copy
import os
import random
import json
import glob
# import ipdb
import sys
import numpy as np 
import pickle 
import torch
import yaml
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pylab as  plt 
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from skimage.morphology import dilation, binary_dilation
from skimage.filters import threshold_otsu

#local
from dataLoad import omnidata
from models import nets
from util import utils
from util.utils import mkdir, weights_init, truncated_z_sample
# from util.csv_logger import CSVLogger, plot_csv
from fid_score import calculate_fid_given_model, get_model


#global variable
real_label = 1.
fake_label = 0.
criterion = nn.BCELoss()

def rescaleImg(x, args, num=0, binary=False, dile=False):

    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((args.imageSize,args.imageSize)), 
                transforms.ToTensor()])

    if (num == 0):
        num = args.batchSize*args.sample_size
    x_list = torch.zeros(num,args.imageSize,args.imageSize)
    for i in range(num):
        x_list[i,:,:] = transform(x[i,:,:])
    x_list = 2 * x_list - 1


    if (dile):
        #randomly dilate images
        for i in range(num):
            toDilate = np.random.choice([0, 1])
            img = x_list[i,:,:]
            if (toDilate):
                img = dilation(img.numpy())
                x_list[i,:,:] = torch.from_numpy(img)
    return x_list

def save_checkpoint(ckpt_name, state, args):
    torch.save(state, os.path.join(args.output_dir, ckpt_name))

def maybe_load_checkpoint(args):
    ckpt_path = os.path.join(args.output_dir, 'ISSA_current.pt')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        return checkpoint

def repeat_context(c, sample_size):
    N = c.shape[0]
    c = c.unsqueeze(1).repeat(1, sample_size, 1)
    c = c.view(N*sample_size, -1)
    return c

def ISSA(train_loader, netG, netD, netE, optimizerG, optimizerD, optimizerE, args, epoch):
    device = args.device
    cuda = True if torch.cuda.is_available() else False
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
    batch_num = 0
    batch_size = args.batchSize
    sample_size = args.sample_size
    total_num = len(train_loader)

    # Optim
    for img_batch in train_loader:
        '''
        define which batch to train
        '''
        real_x = img_batch.view(batch_size*sample_size,28,28)   #reshape x
        real_x = rescaleImg(real_x, args, dile=args.dilate)
        real_x = real_x.view(batch_size*sample_size,args.nc,args.imageSize,args.imageSize)   #reshape x
        real_x = real_x.to(device)
        label = torch.full((batch_size*sample_size,), real_label, device=device)

        '''
        1. train discrminator with real images
        '''
        netD.zero_grad()
        real_c = netE(real_x)
        out_D = netD(real_x.view(-1,sample_size,args.imageSize,args.imageSize), real_c.detach())
        output = out_D.repeat(sample_size)
        errD_real = criterion(output, label)
        errD_real.backward()        #discrminator gradient from real samples
        D_x = output.mean().item()
        real_acc = (output >= .5).float().mean().item()



        '''
        2. train discriminator with fake images
        '''
        noise = torch.randn(batch_size * sample_size, args.nz, device=device)
        context_c = repeat_context(real_c.detach(), sample_size)
        fake = netG(noise, context_c)     #generate fake images
        label.fill_(fake_label)
        stacked = fake.view(-1,sample_size,args.imageSize,args.imageSize).detach()
        out_D = netD(stacked, real_c.detach())
        output = out_D.repeat(sample_size)
        errD_fake = criterion(output, label)
        errD_fake.backward()        #discrminator gradient from fake samples
        optimizerD.step()   #take gradient step
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        fake_acc = (output < .5).float().mean().item()
        acc = .5 * (real_acc+fake_acc)
        d_loss = .5 * (errD_fake.item() + errD_real.item())


        '''
        3. train generator and encoder with discrminator decisions
        '''
        netG.zero_grad()
        netE.zero_grad()
        label.fill_(real_label) 
        fake_imgs = netG(noise, repeat_context(real_c, sample_size))
        out_D = netD(fake_imgs.view(-1,sample_size,args.imageSize,args.imageSize), real_c)
        output = out_D.repeat(sample_size)
        errG = criterion(output, label)
        errG.backward(retain_graph= True)     #generator gradient
        D_G_z2 = output.mean().item()
        D_G_z_loss = (D_G_z1 / D_G_z2)


        '''
        4. real sample with fake label
        '''
        label.fill_(fake_label) 
        out_D = netD(real_x.view(-1,sample_size,args.imageSize,args.imageSize), real_c)
        output = out_D.repeat(sample_size)
        errE = criterion(output, label)
        errE.backward()
        optimizerG.step()
        optimizerE.step()

        ## log performance
        if batch_num % args.log_iter_every == 0:
            print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                    % (epoch, args.epochs, batch_num, total_num, errD.data, errG.data, D_x, D_G_z1, D_G_z2))

        #     # Collect fields
        #     stats_dict = {'global_iteration': iteration_logger.time}
        #     for k in iteration_logger.fieldnames:
        #         if k !='global_iteration':
        #             stats_dict[k] = eval(k)
        #
        #     iteration_logger.writerow(stats_dict)
        #
        # iteration_logger.time += 1
        batch_num += 1



def main(args):
    '''
    load dataset
    '''
    omniglot_dir = args.dataroot + "/omniglot"
    
    device = args.device
    if (args.dataset == "omniglot"):
        train_loader, eval_loader, test_loader, train_sample, eval_dataset, test_dataset = omnidata.load_omniglot_sets(omniglot_dir, args, num_sets=args.num_sets, classifier_split=True)
        args.nc = 1
    else:
        raise NotImplementedError("dataset " + args.dataset + " current not supported")

    '''
    define architecture and initialization
    '''
    cuda = True if torch.cuda.is_available() else False
    encoder = nets.Encoder(args)
    generator = nets.Generator(args)
    discriminator = nets.Discriminator(args)
    
    generator.to(device)
    discriminator.to(device)
    encoder.to(device)

    weights_init(generator, args.g_init)
    weights_init(discriminator, args.d_init)
    weights_init(encoder, args.g_init)

    if args.ckptG != '':
        generator.load_state_dict(torch.load(args.ckptG))

    if args.ckptD != '':
        discriminator.load_state_dict(torch.load(args.ckptD))

    if args.ckptE != '':
        encoder.load_state_dict(torch.load(args.ckptE))

    #optimizers
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lrG, betas=(args.beta1, 0.999), weight_decay=args.wd)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(args.beta1, 0.999), weight_decay=args.wd)
    optimizerE = torch.optim.Adam(encoder.parameters(), lr=args.lrE, betas=(args.beta1, 0.999), weight_decay=args.wd) 

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    fixed_noise = torch.randn(train_sample.shape[0], args.nz, device=device)
    # trunc_noise = truncated_z_sample(mnist_test_batch.shape[0], args.nz, truncation=0.5, seed=args.seed)
    # trunc_noise = torch.from_numpy(trunc_noise).float().to(device)

    x_train = train_sample.view(args.batchSize*args.sample_size, 28, 28)
    x_train = rescaleImg(x_train, args, dile=args.dilate)
    x_train = x_train.view(-1,args.nc,args.imageSize,args.imageSize)   #reshape x

    fid_model = get_model(device)
    train_images, train_labels,eval_images, eval_labels,test_images, test_labels = omnidata.load_raw_omniglot(omniglot_dir, args)



    eval_size = 10
    context_train_set = omnidata.evaluation_loader(args, train_images, train_labels, eval_size, top=True)
    all_train_set = omnidata.evaluation_loader(args, train_images, train_labels, eval_size, top=False)

    dims = 4096


    def evaluate(epoch, best_fid):
        encoder.eval()
        generator.eval()
        # Viz
        with torch.no_grad():
            sample_size = args.sample_size

            train_c = encoder(x_train.to(device))
            train_c = repeat_context(train_c, sample_size)
            fake_train = generator(fixed_noise, train_c).detach()
            fake_train = utils.reconstruct_original(fake_train)
            utils.save_test_grid(utils.reconstruct_original(x_train), fake_train, '%s/viz_sample/%03d_train.jpeg' % (args.output_dir,epoch), args.imageSize, n=10, sample_size=args.sample_size)
            
            imgPclass = 10
            fid = calculate_fid_given_model(context_train_set, all_train_set, encoder, generator, device, args, dims=dims, model=fid_model, imgPclass=imgPclass, eval_size=eval_size)
            print ("at epoch " + str(epoch) + " train fid is " + str(fid))
            encoder.train()
            generator.train()

            return fid

    best_fid = 10000
    best_epoch = 0
    fid_history = []

    # Check for ckpt
    ckpt = maybe_load_checkpoint(args)
    if ckpt is not None:
        print("*"*80)
        print("Loading ckpt \n")
        print("*"*80)
        #
        start_epoch = ckpt['epoch']
        optimizerG  =  ckpt['optimizerG']
        optimizerD  =  ckpt['optimizerD']
        optimizerE = ckpt['optimizerE']
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        encoder.load_state_dict(ckpt['encoder'])

        best_ckpt = torch.load(os.path.join(args.output_dir, 'ISSA_best.pt'))
        best_fid = best_ckpt['fid']
        best_epoch = best_ckpt['epoch']
    else:
        start_epoch = 0

    del eval_loader
    del test_loader

    for epoch in range(start_epoch, args.epochs+1):
        print('*'*100)
        print('Beginning of epoch {}'.format(epoch))
        print('*'*100)

        if args.model == 'ISSA':
            train_loader, eval_loader, test_loader, train_sample, eval_dataset, test_dataset = omnidata.load_omniglot_sets(omniglot_dir, args, num_sets=args.num_sets)
            ISSA(train_loader, generator, discriminator, encoder, optimizerG, optimizerD, optimizerE, args, epoch)
        else:
            raise ValueError("unknown option --model" + args.model)

        # Ckpt
        state = {
            "optimizerG": optimizerG,
            "optimizerD": optimizerD,
            "optimizerE": optimizerE,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "encoder": encoder.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint('ISSA_current.pt', state, args)

        # Eval
        if epoch % args.eval_every == 0:
            fid = evaluate(epoch, best_fid)
            if (best_fid > fid):
                state = {
                "optimizerG": optimizerG,
                "optimizerD": optimizerD,
                "optimizerE": optimizerE,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "encoder": encoder.state_dict(),
                "epoch": epoch,
                "fid": fid
                    }
                save_checkpoint('ISSA_best.pt', state, args)
                best_fid = fid
                best_epoch = epoch
                print ("best training fid so far is " + str(best_fid))
                print ("best epoch updated to be " + str(best_epoch))
                print ("saving best model at epoch " + str(epoch))
            if (fid != 10000):
                fid_history.append(fid)

        # iteration_fieldnames = ['global_iteration','d_loss', 'real_acc','fake_acc','acc']
        # iteration_logger = CSVLogger(every=args.log_iter_every,
        #                              fieldnames=iteration_fieldnames,
        #                              filename=os.path.join(args.output_dir, 'iteration_log.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--dataset', default='omniglot', help='omniglot')
    parser.add_argument('--dataroot', type=str, help='Directory where the dataset .pt files are stored')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2) 
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes used for conditioning')
    parser.add_argument('--sample_size', type=int, default=10, help='number of examples in support set')
    parser.add_argument('--num_sets', type=int, default=25, help='number of support sets sampled per class')
    parser.add_argument('--dilate', type=int, default=0, help='to dilate images')
    parser.add_argument('--binary', type=int, default=0, help='sample binarization of images')
    parser.add_argument('--data_aug', type=int, default=0, help='decide if training images are augmented')

    
    # Model arguments
    parser.add_argument('--model', required=True, help='ISSA')
    parser.add_argument('--c_dim', type=int, default=1000, help='dimension of context vector c')
    parser.add_argument('--nz', type=int, default=1000, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=400)
    parser.add_argument('--ndf', type=int, default=400)
    parser.add_argument('--g_init', type=str, default='N02')
    parser.add_argument('--d_init', type=str, default='N02')
    parser.add_argument('--g_sn', type=int, default=1)
    parser.add_argument('--num_svs', type=int, default=1)
    parser.add_argument('--g_z_scale', type=float, default=1)
    parser.add_argument('--c_scale', type=float, default=0.1)
    parser.add_argument('--g_conditioning_method', type=str, default='cat', choices=['cat','add','mul'])
    parser.add_argument('--truncate', type=float, default=0., help='use truncation trick when sampling, the threshold')


    # Optimization arguments
    parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrE', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    
    parser.add_argument('--wd', type=float, default=0., help='wd for adam')
    parser.add_argument('--seed', type=int, default=2020, help='manual seed')
    parser.add_argument('--d_noise', type=float, default=0)

    # Checkpointing and Logging arguments
    parser.add_argument('--slurm', type=int, default=0, help='is this a slurm job')
    parser.add_argument('--output_dir', required=True, help='directory to save results and weights')
    parser.add_argument('--log_iter_every', type=int, default=1000)
    parser.add_argument('--ckptG', type=str, default='', help='a given checkpoint file for generator')
    parser.add_argument('--ckptD', type=str, default='', help='a given checkpoint file for discriminator')
    parser.add_argument('--ckptE', type=str, default='', help='a given checkpoint file for encoder')
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_ckpt_every', type=int, default=1, help='when to save checkpoint')
    parser.add_argument('--save_imgs_every', type=int, default=1, help='when to save generated images')
    parser.add_argument('--num_gen_images', type=int, default=100, help='number of images to generate for inspection')
    parser.add_argument('--resume', type=int, required=True)
    parser.add_argument('--use_tmp_ckpt_dir', type=int, default=0)
    parser.add_argument('--eval_size', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--viz_details', type=int, default=0)
    parser.add_argument('--eval_only', type=int, default=0)

    args = parser.parse_args()


    print (args.output_dir)
    mkdir(os.path.join(args.output_dir, 'viz_sample'))
    args.jobid = os.environ['SLURM_JOB_ID']
    if (args.slurm != 1):
        utils.save_args(args, os.path.join(args.output_dir, f'args.json'))

    # Global Config
    if not os.path.exists(args.dataroot):
        raise ValueError("can't find dataroot")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device 

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    main(args)