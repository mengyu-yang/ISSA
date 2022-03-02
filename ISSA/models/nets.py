#external
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
from torch.autograd import Variable

#local
from . import layers


class Generator(nn.Module):
    def __init__(self, args, nc=-1):
        super(Generator, self).__init__()
        self.nz = args.nz
        self.ngf = args.ngf
        if (nc == -1):
            self.nc = args.nc
        else:
            self.nc = nc
        self.c_dim = args.c_dim
        self.imgSize = args.imageSize
        self.z_scale = args.g_z_scale
        self.c_scale = args.c_scale
        if (args.g_sn):
            self.which_conv = functools.partial(layers.SNConvTranspose2d, num_svs=args.num_svs, num_itrs=1, eps=1e-12)
        else:
            self.which_conv = nn.ConvTranspose2d

        #architecture for 32*32 images
        if (self.imgSize == 32):
            self.main = nn.Sequential(
                self.which_conv(self.nz + self.c_dim, self.ngf, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                self.which_conv(self.ngf, int(self.ngf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ngf/2)),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                self.which_conv( int(self.ngf/2), int(self.ngf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ngf/4)),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                self.which_conv(int(self.ngf/4), self.nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        elif (self.imgSize == 64):
            self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    self.which_conv(self.nz + self.c_dim, self.ngf, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(self.ngf),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    self.which_conv(self.ngf, int(self.ngf/2), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(self.ngf/2)),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    self.which_conv(int(self.ngf/2), int(self.ngf/4), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(self.ngf/4)),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    self.which_conv( int(self.ngf/4), int(self.ngf/8), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(self.ngf/8)),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    self.which_conv( int(self.ngf/8), self.nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )
        elif (self.imgSize == 128):
            self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    self.which_conv(self.nz + self.c_dim, self.ngf, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(self.ngf),
                    nn.ReLU(True),
                    self.which_conv(self.ngf, int(self.ngf/2), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(self.ngf/2)),
                    nn.ReLU(True),
                    self.which_conv(int(self.ngf/2), int(self.ngf/4), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(self.ngf/4)),
                    nn.ReLU(True),
                    self.which_conv( int(self.ngf/4), int(self.ngf/8), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(self.ngf/8)),
                    nn.ReLU(True),
                    self.which_conv( int(self.ngf/8), int(self.ngf/16), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(int(self.ngf/16)),
                    nn.ReLU(True),
                    self.which_conv( int(self.ngf/16), self.nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )
        else:
            raise NotImplementedError('ImageSize ' + str(self.imgSize) + ' not supported yet.')

    def forward(self, z, c):
        N = z.size(0)
        z = self.z_scale * z
        c = self.c_scale * c
        cat_input = torch.cat((z,c), 1)     #1. concatenate label embedding and image to produce input
        cat_input = cat_input.view(N, -1, 1, 1)
        img = self.main(cat_input)
        img = img.view(N, *(self.nc, self.imgSize, self.imgSize)) #reshape into image dimension
        return img


class NoPacDiscriminator(nn.Module):
    def __init__(self, args, nc=-1):
        super(NoPacDiscriminator, self).__init__()
        self.ndf = args.ndf
        if (nc == -1):
            self.nc = args.nc * args.sample_size
        else:
            self.nc = nc * args.sample_size
        self.imgSize = args.imageSize

        if (args.g_sn):
            self.which_conv = functools.partial(layers.SNConv2d, num_svs=args.num_svs, num_itrs=1, eps=1e-12)
        else:
            self.which_conv = nn.Conv2d

        self.last = nn.Sequential(
            self.which_conv(self.ndf, 1, 4, 1, 0, bias=False)
        )

        if (self.imgSize == 32):
            self.main = nn.Sequential(
                self.which_conv((self.nc), int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/2), self.ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif (self.imgSize == 64):
            self.main = nn.Sequential(
                self.which_conv(self.nc, int(self.ndf/8), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/8)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/8), int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/2), int(self.ndf), 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif (self.imgSize == 128):
            self.main = nn.Sequential(
                self.which_conv(self.nc, int(self.ndf/16), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/16)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/16), int(self.ndf/8), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/8)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/8), int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/2), int(self.ndf), 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True),
            )

        else:
            raise NotImplementedError('ImageSize ' + str(self.imgSize) + ' not supported yet.')

    def out(self, h):
        return self.last(h).sum([2,3])

    '''
    images will come in stacked
    '''
    def forward(self, img):
        h = self.main(img)
        o = self.out(h).view(img.size(0),1)
        o = torch.sigmoid(o)
        o = o.view(-1, 1).squeeze(1)
        return o


class Discriminator(nn.Module):
    def __init__(self, args, nc=-1):
        super(Discriminator, self).__init__()
        self.ndf = args.ndf
        self.c_dim = args.c_dim
        if (nc == -1):
            self.nc = args.nc * args.sample_size
        else:
            self.nc = nc * args.sample_size
        self.imgSize = args.imageSize

        if (args.g_sn):
            self.which_conv = functools.partial(layers.SNConv2d, num_svs=args.num_svs, num_itrs=1, eps=1e-12)
        else:
            self.which_conv = nn.Conv2d

        self.last = nn.Sequential(
            self.which_conv(self.ndf, 1, 4, 1, 0, bias=False)
        )

        self.transform = nn.Sequential(
            nn.Linear(self.c_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.imgSize * self.imgSize)
        )


        if (self.imgSize == 32):
            self.main = nn.Sequential(
                self.which_conv((self.nc+1), int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/2), self.ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif (self.imgSize == 64):
            self.main = nn.Sequential(
                self.which_conv(self.nc+1, int(self.ndf/8), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/8)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/8), int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/2), int(self.ndf), 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif (self.imgSize == 128):
            self.main = nn.Sequential(
                self.which_conv(self.nc+1, int(self.ndf/16), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/16)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/16), int(self.ndf/8), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/8)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/8), int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                self.which_conv(int(self.ndf/2), int(self.ndf), 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True),
            )

        else:
            raise NotImplementedError('ImageSize ' + str(self.imgSize) + ' not supported yet.')

    #sum the output from conv filter
    def cond(self, h):
        return h.sum([2,3])

    def out(self, h):
        return self.last(h).sum([2,3])


    def c_bias(self, c):
        N = c.size(0)
        c = self.transform(c)   #transform c
        c = c.view(N,1,self.imgSize,self.imgSize)
        return c

    '''
    images will come in stacked
    '''
    def forward(self, img, c):
        N = img.size(0)
        c = self.c_bias(c)
        cat_input = torch.cat((img, c), 1)
        h = self.main(cat_input)
        z = self.cond(h)
        o = self.out(h).view(img.size(0),1)
        o = torch.sigmoid(o)
        o = o.view(-1, 1).squeeze(1)
        return o


class Encoder(nn.Module):
    def __init__(self, args, nc=-1):
        super(Encoder, self).__init__()
        self.imgSize = args.imageSize
        self.sample_size = args.sample_size
        self.c_dim = args.c_dim
        self.ndf = args.ndf
        if (nc==-1):
            self.nc = args.nc
        else:
            self.nc = nc

        if (self.imgSize == 32):
            self.main = nn.Sequential(
                nn.Conv2d(self.sample_size*self.nc, int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(int(self.ndf/2), self.ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif(self.imgSize == 64):
            self.main = nn.Sequential(
                nn.Conv2d(self.sample_size*self.nc, int(self.ndf/8), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/8)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(self.ndf/8), int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(self.ndf/2), int(self.ndf), 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif (self.imgSize == 128):
            self.main = nn.Sequential(
                nn.Conv2d(self.sample_size*self.nc, int(self.ndf/16), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/16)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(self.ndf/16), int(self.ndf/8), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/8)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(self.ndf/8), int(self.ndf/4), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/4)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(self.ndf/4), int(self.ndf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(self.ndf/2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(int(self.ndf/2), int(self.ndf), 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.last = nn.Sequential(
            nn.Linear(16*self.ndf, 2000, bias=False),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Linear(2000, self.c_dim, bias=False),
            nn.BatchNorm1d(self.c_dim)
        )

    '''
    output is (input(0), z)
    '''
    def forward(self, img):
        img = img.view(-1, self.sample_size*self.nc, self.imgSize, self.imgSize)
        N = img.size(0)
        h = self.main(img)
        h = h.view(N, -1)
        c = self.last(h)
        c = c.view(N, self.c_dim)
        return c



