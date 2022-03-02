from __future__ import print_function

#external
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pylab as  plt 
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def train_create(IMAGE_PATH,  output_PATH):

    tr = transforms.Compose([transforms.CenterCrop((128)), 
                            transforms.Resize((64,64)),
                            transforms.ToTensor()
                             ])

    celeba = torchvision.datasets.CelebA(IMAGE_PATH, split='train', target_type='identity', transform=tr, target_transform=None, download=False)
    print(len(celeba))
    data_loader = DataLoader(dataset=celeba, batch_size=1000, shuffle=False, num_workers=2, drop_last=False)
    '''
    save it as ndy array, identity = class
    image shape: (#images, nc, width, height)
    label shape: (#images,)
    '''
    out_images = 1
    out_labels = 1
    idx = 0

    for batch in tqdm(data_loader):
        img = batch[0]
        identity = batch[1]
        if (idx == 0):
            out_images = img
            out_labels = identity
        else:
            out_images = torch.cat((out_images, img), 0)
            out_labels = torch.cat((out_labels, identity), 0)
        idx = idx + 1

    print ("finished loading images, now save images")
    torch.save(out_images, output_PATH + 'celeba64.pt')
    torch.save(out_labels, output_PATH + 'celebalabels64.pt')



def test_save(IMAGE_PATH,  output_PATH):
    
    tr = transforms.Compose([transforms.CenterCrop((128)), 
                            transforms.Resize((64,64)),
                            transforms.ToTensor()
                             ])
    celeba = torchvision.datasets.CelebA(IMAGE_PATH, split='test', target_type='identity', transform=tr, target_transform=None, download=False)
    data_loader = DataLoader(dataset=celeba, batch_size=1000, shuffle=False, num_workers=2, drop_last=False)
    out_images = 1
    out_labels = 1
    idx = 0

    for batch in tqdm(data_loader):
        img = batch[0]
        identity = batch[1]
        if (idx == 0):
            out_images = img
            out_labels = identity
        else:
            out_images = torch.cat((out_images, img), 0)
            out_labels = torch.cat((out_labels, identity), 0)
        idx = idx + 1

    print("finished loading images, now save test images")
    torch.save(out_images, output_PATH + 'Test_celeba64.pt')
    torch.save(out_labels, output_PATH + 'Test_celebalabels64.pt')



if __name__ == '__main__':
    IMAGE_PATH = '/scratch/ssd002/datasets/celeba_pytorch'      #specify your own path
    output_PATH = '/scratch/hdd001/home/mengyu/'     #specify your own path
    train_create(IMAGE_PATH, output_PATH)
    test_save(IMAGE_PATH, output_PATH)