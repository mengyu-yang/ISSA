# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import math
import statistics
import re
import contextlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
import dnnlib

import random

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to suppress known warnings in torch.jit.trace().

class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter('ignore', category=torch.jit.TracerWarning)
        return self

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size


    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                # print(self.rank, order[i])
                # print(idx)
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
                # rnd.shuffle(order)
            idx += 1


class DistributedSetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size=32, ssize=3, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5, return_label=True):
        assert len(dataset) > 0
        assert num_replicas > 0     #num_replicas is related to number of gpus
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.ssize = ssize
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

        labels = np.array([int(np.argmax(self.dataset.get_label(idx))) for idx in range(len(self.dataset))])
        cids = np.sort(np.unique(labels))
        '''
        now let self.classes be a dictionary, map label to a consequtive range
        '''
        self.classes = {}
        for i in range(len(cids)):
            self.classes[cids[i]] = i

        all_labels = []
        all_idx = []
        for i in range(len(self.classes)):
            all_labels.append([])
        #constructing a 2d list
        for j in range(len(labels)):
            all_labels[self.classes[labels[j]]].append(j)       #class labels starts from 1 not 0
            all_idx.append(j)
        self.all_idx = all_idx
        self.labels = all_labels
        self.num_samples = int(math.ceil(len(self.all_idx) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_per_label = np.array([len(x) for x in self.labels], dtype=int)

        self.rank_labels = {}
        for i in range(self.num_replicas):
            self.rank_labels[i] = random.sample(range(len(self.classes)), 1)[0]

    def __iter__(self):
        '''
        now need to yield a batch of samples
        '''
        interlabel_window = 0
        intralabel_window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            # rnd.shuffle(self.labels)
            # for i in range(len(self.labels)):
            #     rnd.shuffle(self.labels[i])
            rnd.shuffle(self.all_idx)


        idx = 0
        while True:
            if idx % self.num_replicas == self.rank:
                curr_set_step = ((idx-self.rank)/self.num_replicas) % (self.ssize*self.batch_size) + 1

                # if (curr_set_step % self.ssize) == (self.ssize - 1):
                #     new_label = random.sample(range(len(self.classes)), 1)[0]
                #     self.rank_labels[self.rank] = new_label

                ix = self.labels[self.rank_labels[self.rank]]
                yield random.sample(ix, 1)[0]


            # i = idx % order.size
            # if idx % self.num_replicas == self.rank:
            #     # print(idx)
            #     yield order[i]
            # if window >= 2:
            #     j = (i - rnd.randint(window)) % order.size
            #     order[i], order[j] = order[j], order[i]
            #     # rnd.shuffle(order)
            # idx += 1


        # self.all_idx += self.all_idx[:(self.total_size-len(self.all_idx))]
        # assert len(self.all_idx) == self.total_size
        #
        # self.all_idx = self.all_idx[self.rank:self.total_size:self.num_replicas]
        # assert len(self.all_idx) == self.num_samples
        #
        # while True:
        #     subset = random.sample(self.all_idx, 1)[0]
        #     yield subset

        #
        # idx_interlabel = 0
        # idx_intralabel = 0
        # while True:
        #     out_idx = []
        #     j = idx_interlabel % len(self.all_idx)
        #     if idx_interlabel % self.num_replicas == self.rank:
        #         yield self.all_idx[j:j+self.ssize*self.batch_size]
        #         # for indices in candidates:
        #         #     out_idx.extend(indices[:self.ssize])
        #         # print(out_idx)
        #
        #     idx_interlabel += 1


'''
write an iterator which samples sets with the same labels
'''
class SetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size=32, ssize=3, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5, return_label=True):
        assert len(dataset) > 0
        assert num_replicas > 0     #num_replicas is related to number of gpus
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.ssize = ssize
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        labels = np.array([int(np.argmax(self.dataset.get_label(idx))) for idx in range(len(self.dataset))])
        cids = np.sort(np.unique(labels))
        '''
        now let self.classes be a dictionary, map label to a consequtive range
        '''
        self.classes = {}
        for i in range(len(cids)):
            self.classes[cids[i]] = i

        all_labels = []
        for i in range(len(self.classes)):
            all_labels.append([])
        #constructing a 2d list
        for j in range(len(labels)):
            all_labels[self.classes[labels[j]]].append(j)       #class labels starts from 1 not 0
        self.labels = all_labels

    def sample_set(self, cid):

        ix = self.labels[cid]     #remember that labels starts from 1 and idx starts from 0
        num_instances = len(ix)
        while (num_instances < self.ssize):
            # raise Exception('class' + str(cid) + " has less than" + str(self.ssize) + " images")
            # resample
            cid = random.sample(range(len(self.classes)), 1)[0]
            ix = self.labels[cid]  # remember that labels starts from 1 and idx starts from 0
            num_instances = len(ix)

        return random.sample(ix, self.ssize)


    '''
    do 2 things
    1. order the labels based on sets
    return mini_batch of images
    2. endless loop provides idx
    3. randomly sample the classes
    '''
    def __iter__(self):
        '''
        now need to yield a batch of samples
        '''
        idx = 0
        while True:
            if idx % self.num_replicas == self.rank:
                out_idx = []
                k = 0
                cids = random.sample(range(len(self.classes)), self.batch_size)
                while k < self.batch_size:
                    cid = cids[k]
                    out_idx = out_idx + self.sample_set(cid)
                    k += 1
                # for j in range(len(out_idx)):
                #     labels = [int(np.argmax(self.dataset.get_label(idx))) for idx in out_idx]
                # print(labels)
                # print(self.rank, len(out_idx))
                yield out_idx
            idx += 1
#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        # print(name, tensor.shape, other.shape, torch.unique(tensor-other))
        assert (nan_to_num(tensor) == nan_to_num(other)).all(), fullname

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs

#----------------------------------------------------------------------------

def plot_grad_flow(named_parameters, fname):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(fname)

def get_grad(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())

    return statistics.mean(ave_grads)