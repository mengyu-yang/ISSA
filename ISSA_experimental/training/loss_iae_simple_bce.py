# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, use_softmax=True, shuffledisc=True, concatlabel=False, augment_pipe=None, E=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.E = E
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.encode = False
        if self.E is not None: self.encode = True

        self.softmax = nn.Softmax(dim=1)
        self.use_softmax = use_softmax
        self.shuffledisc = shuffledisc
        self.concatlabel = concatlabel

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.real_label = 1.
        self.fake_label = 0.



    def run_E(self, img, sync):
        with misc.ddp_sync(self.E, sync):
            with torch.no_grad():
                context, ssize, feature_map = self.E(img)
                context = context.repeat_interleave(ssize, dim=0)
        return context, feature_map, ssize

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G, sync):
            img = self.G(z, c)
            # print('G out: ', img.shape)
        return img

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            # print('D_in shape: ', img.shape)
            logits = self.D(img, c)
            # print('D out: ', logits.shape)
        return logits

    def run_D_shuffle(self, img, c, ssize, sync, ssize_mult=1):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            # img: (ssize*numsets, 3, 64, 64)

            total_logits = 0
            img = img.view(-1, ssize, 3, 64, 64)
            for i in range(ssize*ssize_mult):
                perm_idx = torch.randperm(ssize)
                img_perm = img[:, perm_idx, :, :, :]
                img_perm = img_perm.view(-1, 3, 64, 64)
                # print('D_in shape: ', img.shape)
                logits = self.D(img_perm, c)
                total_logits += logits
                # print('D out: ', logits.shape)
        total_logits /= ssize
        return total_logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                if self.encode:
                    context, feature_map, _ = self.run_E(real_img, sync=(sync and not do_Gpl))
                    gen_img = self.run_G(gen_z, context, sync=(sync and not do_Gpl))  # May get synced by Gpl.
                    gen_logits = self.run_D(gen_img, context, sync=False)
                else:
                    gen_img = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                    gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                # loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                label = torch.full_like(gen_logits, self.real_label, dtype=torch.float, device=self.device)
                loss_Gmain = self.criterion(gen_logits, label)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()


        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                if self.encode:
                    context, feature_map, _ = self.run_E(real_img, sync=False)
                    gen_img = self.run_G(gen_z, context, sync=False)
                    gen_logits = self.run_D(gen_img, context, sync=False)  # Gets synced by loss_Dreal.
                else:
                    gen_img = self.run_G(gen_z, gen_c, sync=False)
                    gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                # loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                label = torch.full_like(gen_logits, self.fake_label, dtype=torch.float, device=self.device)
                loss_Dgen = self.criterion(gen_logits, label)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain:
            name = 'Dreal'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                if self.encode:
                    context, feature_map, ssize = self.run_E(real_img, sync=sync)
                    real_logits = self.run_D(real_img_tmp, context, sync=sync)

                else:
                    real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                # loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                label = torch.full_like(real_logits, self.real_label, dtype=torch.float, device=self.device)
                loss_Dreal = self.criterion(real_logits, label)
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
