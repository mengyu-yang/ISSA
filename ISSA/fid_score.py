"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
code adapted from https://github.com/mseitzer/pytorch-fid
"""
#external imports
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import cpu_count
import math

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.utils import save_image
from io_utils import model_dict
from util import utils
from util.utils import mkdir, truncated_z_sample
from inception import InceptionV3



try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

#local imports
from models import nets, omni_classifier_all

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dataset', type=str, default="omniglot",
                    help='omniglot')
parser.add_argument('--dims', type=int, default=4096,
                    help=('Dimensionality of classifer features to use. '
                          'By default, uses pool3 features'))

parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu', imageSize=32, flip=False):

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)


    transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((imageSize,imageSize)),
                transforms.ToTensor()])
    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=cpu_count())

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        if (flip):
            batch = 1 - batch
        batch = batch.to(device)
        N = batch.size(0)
        with torch.no_grad():
            if (callable(model)):
                pred = model.forward(batch)
            else:
                pred = model(batch)

        pred = pred.view(N,-1,1,1)
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=4096,
                                    device='cpu', imageSize=32, flip=False):
    act = get_activations(files, model, batch_size, dims, device, imageSize=imageSize, flip=flip)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, imageSize=32):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, imageSize=imageSize, flip=False)

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, dataset="omniglot"):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if (dataset=="omniglot"):
        model = omni_classifier_all.Net()
        model.load_state_dict(torch.load(os.getcwd()+"/omni_all_newsplit.pth"))
        model.to(device)
        model.eval()
        imageSize = 32

    elif(dataset == "celeba"):
        dims=2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        imageSize = 64


    else:
        raise Exception("dataset not implemented for fid evaluation yet")

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, imageSize=imageSize)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, imageSize=imageSize)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_model(device, dataset="omniglot"):

    if (dataset == "omniglot"):
        model = omni_classifier_all.Net()
        model.load_state_dict(torch.load(os.getcwd()+"/omni_all_newsplit.pth"))
        model.to(device)
        model.eval()
    elif (dataset == "celeba"):
        dims=2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        
    return model

def calculate_fid_given_model_nocond(eval_loader, train_loader, generator, device, args, dims=64, model=None, imgPclass=20, eval_size=10, num_support=-1):
    if (model is None):
        if (args.dataset == "omniglot"):
            model = omni_classifier_all.Net()
            model.load_state_dict(torch.load(os.getcwd()+"/omni_all_newsplit.pth"))
            model.to(device)
            model.eval()
            imageSize = 32

    m1, s1 = get_real_statistics(train_loader, model, device, args, imgPclass, dims=dims)

    m2, s2 = get_fake_statistics_nocond(eval_loader, generator, model, device, args, imgPclass, dims=dims, eval_size=eval_size, num_support=num_support)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def calculate_fid_given_model(eval_loader, train_loader, encoder, generator, device, args, dims=64, model=None, imgPclass=20, eval_size=10, num_support=-1):
    if (model is None):
        if (args.dataset == "omniglot"):
            model = omni_classifier_all.Net()
            model.load_state_dict(torch.load(os.getcwd()+"/omni_all_newsplit.pth"))
            model.to(device)
            model.eval()
            imageSize = 32

    m1, s1 = get_real_statistics(train_loader, model, device, args, imgPclass, dims=dims)

    m2, s2 = get_fake_statistics(eval_loader, encoder, generator, model, device, args, imgPclass, dims=dims, eval_size=eval_size, num_support=num_support)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fid_celeba(eval_loader, train_loader, train_images, encoder, generator, device, args, dims=64, model=None, imgPclass=20, eval_size=10, num_support=-1):
    if (model is None):
        if (args.dataset == "celeba"):
            dims=2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx]).to(device)
            model.eval()
            imageSize = 64

    m1, s1 = get_real_statistics_celeba(train_loader, train_images, model, device, args)

    m2, s2 = get_fake_statistics_celeba(eval_loader, train_images, encoder, generator, model, device, args, imgPclass, dims=dims, eval_size=eval_size, num_support=num_support)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def calculate_fid_celeba_label(eval_loader, train_loader, train_images, train_labels, generator, device, args, dims=64, model=None, imgPclass=20, eval_size=10, num_support=-1):
    if (model is None):
        if (args.dataset == "celeba"):
            dims=2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx]).to(device)
            model.eval()
            imageSize = 64

    m1, s1 = get_real_statistics_celeba_label(train_loader, train_images, model, device, args)

    m2, s2 = get_fake_statistics_celeba_label(eval_loader, train_images, train_labels, generator, model, device, args, imgPclass, dims=dims, eval_size=eval_size, num_support=num_support)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def get_real_statistics(train_loader, model, device, args, imgPclass, dims=64):
    real_arr = np.empty((len(train_loader.dataset)*imgPclass, dims))
    start_idx = 0

    to32 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32,32)), 
                transforms.ToTensor()])

    for batch in train_loader:

        batch = batch.view(-1,28,28)
        N = batch.size(0)

        batch = batch.view(-1,1,28,28)
        real_x = torch.zeros(batch.size(0),32,32)
        for i in range(batch.size(0)):
            real_x[i,:,:] = to32(batch[i,:,:])

        real_x = real_x.view(-1,1,32,32)
        real_x = real_x.to(device)
        with torch.no_grad():
            if (callable(model)):
                pred = model.forward(real_x)
            else:
                pred = model(real_x)

        pred = pred.view(N,-1,1,1)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        real_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    m1 = np.mean(real_arr, axis=0)
    s1 = np.cov(real_arr, rowvar=False)

    return m1, s1


def get_real_statistics_celeba(train_loader, train_images, model, device, args):
    real_arrs = []
    start_idx = 0

    for batch in train_loader:
        batch = train_images[batch]
        N = batch.size(0)

        real_x = batch.view(-1,args.nc,args.imageSize,args.imageSize)
        real_x = real_x.to(device)
        with torch.no_grad():
            if (callable(model)):
                pred = model.forward(real_x)
            else:
                pred = model(real_x)
        if isinstance(pred, list):
            pred = pred[0]

        pred = pred.view(N,-1,1,1)
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        real_arrs.append(pred)
        start_idx = start_idx + pred.shape[0]

    real_arr = np.concatenate(real_arrs, axis=0)
    m1 = np.mean(real_arr, axis=0)
    s1 = np.cov(real_arr, rowvar=False)

    return m1, s1

def get_real_statistics_celeba_label(train_loader, train_images, model, device, args):
    real_arrs = []
    start_idx = 0

    for batch in train_loader:
        img_idx = batch[0]
        img_batch = train_images[img_idx]
        N = img_batch.size(0)

        real_x = img_batch.view(-1,args.nc,args.imageSize,args.imageSize)
        real_x = real_x.to(device)
        with torch.no_grad():
            if (callable(model)):
                pred = model.forward(real_x)
            else:
                pred = model(real_x)
        if isinstance(pred, list):
            pred = pred[0]

        pred = pred.view(N,-1,1,1)
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        real_arrs.append(pred)
        start_idx = start_idx + pred.shape[0]

    real_arr = np.concatenate(real_arrs, axis=0)
    m1 = np.mean(real_arr, axis=0)
    s1 = np.cov(real_arr, rowvar=False)

    return m1, s1


def repeat_context(c, sample_size):
    N = c.shape[0]
    c = c.unsqueeze(1).repeat(1, sample_size, 1)
    c = c.view(N*sample_size, -1)
    return c


def get_fake_statistics_nocond(eval_loader, generator, model, device, args, imgPclass, dims=64, eval_size=10, num_support=-1):
    fake_arr = np.empty((len(eval_loader.dataset)*imgPclass, dims))

    generator.eval()


    start_idx = 0
    to32 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32,32)),
                transforms.ToTensor()])

    to28 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((28,28)),
                transforms.ToTensor()])

    for batch in eval_loader:
        #convert image to 32*32
        batch = batch.view(-1,28,28)
        N = batch.size(0)
        real_x = torch.zeros(batch.size(0),32,32)
        for i in range(batch.size(0)):
            real_x[i,:,:] = to32(batch[i,:,:])

        real_x = real_x.view(-1,1,32,32)
        real_x = 2 * real_x - 1     #normalize to [-1,1] scale for tanh activation to be consistent
        if (num_support != -1):
            real_x = real_x.view(-1,num_support,32,32)
            real_x = torch.repeat_interleave(real_x, int(args.sample_size/num_support), dim=1)
        real_x = real_x.view(-1,args.sample_size,32,32)

        if (num_support != -1):
            gen_size = int(imgPclass / (int(eval_size/num_support)))     #number of images generated per context
        else:
            gen_size = int(imgPclass / (int(eval_size/args.sample_size)))     #number of images generated per context
        fake_gen = torch.zeros(real_x.size(0),gen_size,32,32)

        for i in range(real_x.size(0)):
            real_input = real_x[i]
            real_input = real_input.view(args.sample_size,32,32)
            with torch.no_grad():
                N = real_input.size(0)

                z_sample = torch.randn(1, 512, device=device)
                fake_train,_ = generator([z_sample])
                fake_gen[i] = fake_train.detach().view(-1,32,32)

        fake_gen = fake_gen.view(-1,1,32,32)
        fake_gen = fake_gen.to(device)
        fake_gen = utils.reconstruct_orig(fake_gen, drange=[-1, 1])

        fake_pred = model.forward(fake_gen)
        fake_pred = fake_pred.view(fake_gen.size(0),-1,1,1)
        fake_pred = fake_pred.squeeze(3).squeeze(2).cpu().detach().numpy()
        fake_arr[start_idx:start_idx + fake_pred.shape[0]] = fake_pred
        start_idx = start_idx + fake_pred.shape[0]

    m2 = np.mean(fake_arr, axis=0)
    s2 = np.cov(fake_arr, rowvar=False)

    generator.train()
    return m2, s2


def get_fake_statistics(eval_loader, encoder, generator, model, device, args, imgPclass, dims=64, eval_size=10, num_support=-1):
    fake_arr = np.empty((len(eval_loader.dataset)*imgPclass, dims))

    encoder.eval()
    generator.eval()


    start_idx = 0
    to32 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32,32)), 
                transforms.ToTensor()])

    to28 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((28,28)), 
                transforms.ToTensor()])

    for batch in eval_loader:
        #convert image to 32*32
        batch = batch.view(-1,28,28)
        N = batch.size(0)
        real_x = torch.zeros(batch.size(0),32,32)
        for i in range(batch.size(0)):
            real_x[i,:,:] = to32(batch[i,:,:])

        real_x = real_x.view(-1,1,32,32)
        real_x = 2 * real_x - 1     #normalize to [-1,1] scale for tanh activation to be consistent
        if (num_support != -1):
            real_x = real_x.view(-1,num_support,32,32)
            real_x = torch.repeat_interleave(real_x, int(args.sample_size/num_support), dim=1)
        real_x = real_x.view(-1,args.sample_size,32,32)

        if (num_support != -1):
            gen_size = int(imgPclass / (int(eval_size/num_support)))     #number of images generated per context
        else:
            gen_size = int(imgPclass / (int(eval_size/args.sample_size)))     #number of images generated per context
        fake_gen = torch.zeros(real_x.size(0),gen_size,32,32)

        for i in range(real_x.size(0)):
            real_input = real_x[i]
            real_input = real_input.view(args.sample_size,32,32)
            with torch.no_grad():
                N = real_input.size(0)
                train_c = encoder(real_input .to(device))
                if (train_c.size(0) < N):
                    train_c = repeat_context(train_c, args.sample_size)
                train_c = train_c.view(-1,args.sample_size,args.c_dim)
                context = train_c[:,0,:]        #the first context in each set
                context = context.view(-1,args.c_dim)
                context = torch.repeat_interleave(context, gen_size, dim=0)
                noise = torch.randn(context.size(0), args.nz, device=device)

                fake_train = generator(noise, context).detach()
                fake_gen[i] = fake_train.view(-1,32,32)

        fake_gen = fake_gen.view(-1,1,32,32)
        fake_gen = fake_gen.to(device)
        fake_gen = utils.reconstruct_orig(fake_gen, drange=[-1, 1])

        fake_pred = model.forward(fake_gen)
        fake_pred = fake_pred.view(fake_gen.size(0),-1,1,1)
        fake_pred = fake_pred.squeeze(3).squeeze(2).cpu().detach().numpy()
        fake_arr[start_idx:start_idx + fake_pred.shape[0]] = fake_pred
        start_idx = start_idx + fake_pred.shape[0]

    m2 = np.mean(fake_arr, axis=0)
    s2 = np.cov(fake_arr, rowvar=False)

    encoder.train()
    generator.train()
    return m2, s2

def get_fake_statistics_celeba_label(eval_loader, train_images, train_labels, generator, model, device, args, imgPclass, dims=64, eval_size=10, selected=False, num_support=-1):
    fake_arrs = []

    generator.eval()
    start_idx = 0

    for batch in eval_loader:
        img_idx, label_idx = batch[0], batch[1]
        batch_img = train_images[img_idx]
        batch_label = train_labels[label_idx]

        real_x = batch_img.view(-1,args.nc,args.imageSize,args.imageSize)

        if (not args.selected):
            real_x = 2 * real_x - 1     #normalize to [-1,1] scale for tanh activation to be consistent

        if (num_support != -1):
            real_x = real_x.view(-1,num_support,args.nc,args.imageSize,args.imageSize)
            real_x = torch.repeat_interleave(real_x, int(args.sample_size/num_support), dim=1)

        real_x = real_x.view(-1,args.sample_size*args.nc,args.imageSize,args.imageSize)
        if (num_support != -1):
            gen_size = int(imgPclass / (int(eval_size/num_support)))     #number of images generated per context
        else:
            gen_size = int(imgPclass / (int(eval_size/args.sample_size)))     #number of images generated per context
        fake_gen = torch.zeros(real_x.size(0),gen_size,args.nc,args.imageSize,args.imageSize)

        for i in range(real_x.size(0)):
            real_input = real_x[i]
            real_input = real_input.view(args.sample_size*args.nc,args.imageSize,args.imageSize)
            with torch.no_grad():
                N = real_input.size(0)
                train_c = F.one_hot(batch_label, num_classes=10178).float().to(device)
                if (train_c.size(0) < N):
                    train_c = repeat_context(train_c, args.sample_size)
                train_c = train_c.view(-1,args.sample_size,args.c_dim)
                context = train_c[:,0,:]        #the first context in each set
                context = context.view(-1,args.c_dim)
                context = torch.repeat_interleave(context, gen_size, dim=0)
                noise = torch.randn(context.size(0), args.nz, device=device)
                fake_train = generator(noise, context).detach()
                fake_gen[i] = fake_train.view(gen_size,args.nc,args.imageSize,args.imageSize)

        fake_gen = fake_gen.view(-1,args.nc,args.imageSize,args.imageSize)
        fake_gen = fake_gen.to(device)
        fake_gen = utils.reconstruct_orig(fake_gen, drange=[-1, 1])

        fake_pred = model.forward(fake_gen)
        if isinstance(fake_pred, list):
            fake_pred = fake_pred[0]
        fake_pred = fake_pred.view(fake_gen.size(0),-1,1,1)
        fake_pred = fake_pred.squeeze(3).squeeze(2).cpu().detach().numpy()
        fake_arrs.append(fake_pred)
        start_idx = start_idx + fake_pred.shape[0]

    fake_arr = np.concatenate(fake_arrs, axis=0)
    m2 = np.mean(fake_arr, axis=0)
    s2 = np.cov(fake_arr, rowvar=False)

    generator.train()
    return m2, s2


def get_fake_statistics_celeba(eval_loader, train_images, encoder, generator, model, device, args, imgPclass, dims=64, eval_size=10, selected=False, num_support=-1):
    fake_arrs = []

    encoder.eval()
    generator.eval()
    start_idx = 0

    for batch in eval_loader:
        batch = train_images[batch]

        real_x = batch.view(-1,args.nc,args.imageSize,args.imageSize)

        if (not args.selected):
            real_x = 2 * real_x - 1     #normalize to [-1,1] scale for tanh activation to be consistent

        if (num_support != -1):
            real_x = real_x.view(-1,num_support,args.nc,args.imageSize,args.imageSize)
            real_x = torch.repeat_interleave(real_x, int(args.sample_size/num_support), dim=1)

        real_x = real_x.view(-1,args.sample_size*args.nc,args.imageSize,args.imageSize)
        if (num_support != -1):
            gen_size = int(imgPclass / (int(eval_size/num_support)))     #number of images generated per context
        else:
            gen_size = int(imgPclass / (int(eval_size/args.sample_size)))     #number of images generated per context
        fake_gen = torch.zeros(real_x.size(0),gen_size,args.nc,args.imageSize,args.imageSize)

        for i in range(real_x.size(0)):
            real_input = real_x[i]
            real_input = real_input.view(args.sample_size*args.nc,args.imageSize,args.imageSize)
            with torch.no_grad():
                N = real_input.size(0)
                train_c = encoder(real_input.to(device))
                if (train_c.size(0) < N):
                    train_c = repeat_context(train_c, args.sample_size)
                train_c = train_c.view(-1,args.sample_size,args.c_dim)
                context = train_c[:,0,:]        #the first context in each set
                context = context.view(-1,args.c_dim)
                context = torch.repeat_interleave(context, gen_size, dim=0)
                noise = torch.randn(context.size(0), args.nz, device=device)
                fake_train = generator(noise, context).detach()
                fake_gen[i] = fake_train.view(gen_size,args.nc,args.imageSize,args.imageSize)

        fake_gen = fake_gen.view(-1,args.nc,args.imageSize,args.imageSize)
        fake_gen = fake_gen.to(device)
        fake_gen = utils.reconstruct_orig(fake_gen, drange=[-1, 1])

        fake_pred = model.forward(fake_gen)
        if isinstance(fake_pred, list):
            fake_pred = fake_pred[0]
        fake_pred = fake_pred.view(fake_gen.size(0),-1,1,1)
        fake_pred = fake_pred.squeeze(3).squeeze(2).cpu().detach().numpy()
        fake_arrs.append(fake_pred)
        start_idx = start_idx + fake_pred.shape[0]

    fake_arr = np.concatenate(fake_arrs, axis=0)
    m2 = np.mean(fake_arr, axis=0)
    s2 = np.cov(fake_arr, rowvar=False)

    encoder.train()
    generator.train()
    return m2, s2


def main():
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          dataset=args.dataset)
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()
