import os 
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'

from scipy.stats import truncnorm
import numpy as  np
# import ipdb
import json
import torch
from torch.nn import init
from torchvision.utils import save_image, make_grid
from types import SimpleNamespace
from tqdm import tqdm 
import matplotlib.pyplot as plt
import re 



class Stats:
    def __init__(self):
        self.items = []
    def add(self, x):
        self.items.append(x)
    def avg(self):
        return np.mean(self.items)
        

def extractMax(input):
    # get a list of all numbers separated by
    # lower case characters
    # \d+ is a regular expression which means
    # one or more digit
    # output will be like ['100','564','365']
    numbers = []
    for i in range(len(input)):
        temp = re.findall('\d+', input[i])
        if len(temp) > 0:
            for j in range(len(temp)):
                temp[j] = int(temp[j])
            numbers.append(max(temp))

    return max(numbers)

def save_args(args, fp):
    # Save config
    config_file = open(fp, "wt")
    config_file.write(str(args))
    config_file.flush()

def load_args(fp):
    return SimpleNamespace(**json.load(open(fp, 'r')))


def generate_n_samples(generator, generator_args, device, N):
    all_samples = []
    for start in tqdm(range(0, N, generator_args.num_gen_images), desc='generate_n_samples'): 
        with torch.no_grad():
            fake = generator(torch.randn(generator_args.num_gen_images, generator_args.nz, 1, 1, device=device))
        all_samples.append(fake)
    return torch.cat(all_samples, 0)
    


def weights_init(model, method='N02'):
    for m in model.modules():
        classname = m.__class__.__name__
        if (isinstance(m, torch.nn.Conv2d)
            or isinstance(m, torch.nn.Linear)
            or isinstance(m, torch.nn.Embedding)):
            print(f'Initializing: {classname}')
            if method == 'ortho':
                init.orthogonal_(m.weight)
            elif method == 'N02':
                init.normal_(m.weight, 0, 0.02)
            elif method in ['glorot', 'xavier']:
                init.xavier_uniform_(m.weight)
            else:
                print('Init style not recognized...')
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def tonp(x):
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()
    return x

def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
    return truncation * values


def save_samples_grid(inputs, samples, save_path, imgSize, n=20, sample_size=5, nc=1):
    inputs = inputs.cpu().data.view(-1, sample_size, nc, imgSize, imgSize)[:n]
    reconstructions = samples.cpu().data.view(-1, 10, nc, imgSize, imgSize)[:n]

    grid_img = inputs.view(-1, nc, imgSize, imgSize)
    grid_input = make_grid(grid_img, padding=2, nrow=sample_size, pad_value=1)
    grid_img = reconstructions.view(-1, nc, imgSize, imgSize)
    grid_fake = make_grid(grid_img, padding=2, nrow=10, pad_value=1)
    save_image(grid_input, save_path + '_input.jpeg', normalize=True, nrow=1)
    save_image(grid_fake, save_path + '.jpeg', normalize=True)


def save_test_grid(inputs, samples, save_path, imgSize, n=10, sample_size=5, nc=1):

    inputs = inputs.cpu().data.view(-1, sample_size, nc, imgSize, imgSize)[:n]
    reconstructions = samples.cpu().data.view(-1, 10, nc, imgSize, imgSize)[:n]

    grid_img = inputs.view(-1,nc, imgSize, imgSize)
    grid_input = make_grid(grid_img, padding=2, nrow=sample_size, pad_value=1)
    grid_img = reconstructions.view(-1,nc, imgSize, imgSize)
    grid_fake = make_grid(grid_img, padding=2, nrow=10, pad_value=1)
    grids = make_grid([grid_input, grid_fake], padding=20, nrow=2, pad_value=1)
    save_image(grids, save_path, normalize=True, nrow=1)
    # save_image(grid_fake, save_path, normalize=True)

def save_sidebyside(inputs, samples, save_path, imgSize, n=10, nc=1):
    inputs = inputs.cpu().data.view(-1, 5, nc, imgSize, imgSize)[:n]
    reconstructions = samples.cpu().data.view(-1, 5, nc, imgSize, imgSize)[:n]
    images = inputs[:,0].view(-1, 1, nc, imgSize, imgSize)
    for i in range(reconstructions.size(1)-1):
        images = torch.cat((images, reconstructions[:,i].view(-1, 1, nc, imgSize, imgSize)), dim=1)
        images = torch.cat((images, inputs[:,i+1].view(-1, 1, nc, imgSize, imgSize)), dim=1)
    images = torch.cat((images, reconstructions[:,reconstructions.size(1)-1].view(-1, 1, nc, imgSize, imgSize)), dim=1).view(-1, nc, imgSize, imgSize)
    save_image(images, save_path, normalize=True, nrow=n)

def save_diversity(inputs, samples, save_path, imgSize, nc=1):
    inputs = inputs.cpu().data.view(-1, nc, imgSize, imgSize)        #real image
    reconstructions = samples.cpu().data.view(-1, 10, nc, imgSize, imgSize)
    images = inputs[0].view(1, nc, imgSize, imgSize)

    for i in range(reconstructions.size(0)-1):
        images = torch.cat((images, reconstructions[i].view(10, nc, imgSize, imgSize)), dim=0)
        images = torch.cat((images, inputs[i+1].view(1, nc, imgSize, imgSize)), dim=0)
    images = torch.cat((images, reconstructions[-1].view(10, nc, imgSize, imgSize)), dim=0)
    images = images.view(-1, nc, imgSize, imgSize)
    save_image(images, save_path, normalize=True, nrow=11)


def reconstruct_orig(x, drange):
    lo, hi = drange
    x = (x - lo) * (255 / (hi - lo))
    x = torch.clamp(x.round(), min=0, max=255)
    return x

def original_ISSA_reconstruct(x):
    x = (x + 1) / 2
    return x




