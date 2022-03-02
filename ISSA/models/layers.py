import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P



def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x



def power_iteration(W, u_, update=True, eps=1e-12):
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    with torch.no_grad():
      v = torch.matmul(u, W)
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      vs += [v]
      u = torch.matmul(v, W.t())
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      us += [u]
      if update:
        u_[i][:] = u
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
  return svs, us, vs


class identity(nn.Module):
  def forward(self, input):
    return input
 

class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    self.num_itrs = num_itrs
    self.num_svs = num_svs
    self.transpose = transpose
    self.eps = eps
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    if self.training:
      with torch.no_grad(): 
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


class SNConvTranspose2d(nn.ConvTranspose2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, out_padding=0, groups=1, bias=True, dilation=1,
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, out_padding, groups, bias, dilation)
    SN.__init__(self, num_svs, num_itrs, in_channels, eps=eps)    
  def forward(self, x):
    return F.conv_transpose2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.output_padding, self.groups, self.dilation)


class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)


class SNEmbedding(nn.Embedding, SN):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None, 
               max_norm=None, norm_type=2, scale_grad_by_freq=False,
               sparse=False, _weight=None,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq, 
                          sparse, _weight)
    SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
  def forward(self, x):
    return F.embedding(x, self.W_())



class myBN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super(myBN, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('stored_mean', torch.zeros(num_channels))
        self.register_buffer('stored_var', torch.ones(num_channels))
        self.register_buffer('accumulation_counter', torch.zeros(1))
        self.accumulate_standing = False

    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            else:
                self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
                self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
            return out
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


def groupnorm(x, norm_style):
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    else:
        groups = 16
    return F.group_norm(x, groups)


class ccbn(nn.Module):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False, norm_style='bn',):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn
        self.norm_style = norm_style

        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
        elif self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var',    torch.ones(output_size))

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        if self.mybn or self.cross_replica:
            return self.bn(x, gain=gain, bias=bias)
        else:
            if self.norm_style == 'bn':
                out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps)
            elif self.norm_style == 'in':
                out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None, self.training, 0.1, self.eps)
            elif self.norm_style == 'gn':
                out = groupnorm(x, self.normstyle)
            elif self.norm_style == 'nonorm':
                out = x
            return out * gain + bias
    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s +=' cross_replica={cross_replica}'
        return s.format(**self.__dict__)


class bn(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1, cross_replica=False, mybn=False):
        super(bn, self).__init__()
        self.output_size= output_size
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.cross_replica = cross_replica
        self.mybn = mybn

        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
        elif mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        else:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var',    torch.ones(output_size))

    def forward(self, x, y=None):
        if self.cross_replica or self.mybn:
            gain = self.gain.view(1,-1,1,1)
            bias = self.bias.view(1,-1,1,1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                                self.bias, self.training, self.momentum, self.eps)


