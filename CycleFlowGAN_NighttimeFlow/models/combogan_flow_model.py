import numpy as np
import torch
import os
import random
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
from PIL import Image
from torchvision import transforms as vtransforms
from torchvision import utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from .multiloss import  *
from util import flow_utils
from skimage.transform import resize

class ComboGANflowModel(BaseModel):
    def name(self):
        return 'ComboGANflowModel'

    def __init__(self, opt):
        super(ComboGANflowModel, self).__init__(opt)

        self.n_domains    = opt.n_domains
        self.DA, self.DB  = None, None
        self.loadSize     = opt.loadSize
        self.cropSize     = opt.cropSize
        self.batchSize    = opt.batchSize
        self.input_nc     = opt.input_nc
        self.use_grayscale_images  = False

        # load/define networks
        self.netG   = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        self.netFlow = networks.define_FlowNet(opt.use_grayscale_images, self.n_domains, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, self.Tensor, opt.norm, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            self.netFlow.load(os.path.join(self.save_dir, '%d_net_%s' % (which_epoch, 'FlowNet0.pth')), \
                              os.path.join(self.save_dir, '%d_net_%s' % (which_epoch, 'FlowNet1.pth')),
                              opt)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)
        else:
            self.netFlow.load(opt.loadmodel_flownet, opt.loadmodel_flownet, opt)

        if self.isTrain:
            self.fake_pools     = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            self.L1             = torch.nn.SmoothL1Loss()
            self.downsample     = torch.nn.AvgPool2d(3, stride=2)
            self.criterionCycle = self.L1
            # self.criterionGAN   = lambda r,f,v : (networks.GANLoss(r[0], f[0], v) + \
            #                                       networks.GANLoss(r[1], f[1], v) + \
            #                                       networks.GANLoss(r[2], f[2], v) + \
            #                                       networks.GANLoss(r[3], f[3], v) + \
            #                                       networks.GANLoss(r[4], f[4], v) + \
            #                                       networks.GANLoss(r[5], f[5], v)) / 6
            self.criterionGAN = lambda r,f,v : (networks.GANLoss(r[0], f[0], v) + \
                                                networks.GANLoss(r[1], f[1], v)) / 2

            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam,    opt.lr, (opt.beta1, 0.999))
            self.netFlow.init_optimizers(torch.optim.Adam, opt.lr/10, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam,    opt.lr, (opt.beta1, 0.999))
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle   = [0]*self.n_domains
            self.loss_flowrec = [0]*self.n_domains
            # initialize loss multipliers
            self.lambda_cyc     = opt.lambda_cycle
            self.lambda_flowrec = opt.lambda_flowrec

        print('---------- Networks initialized ---------------')
        print('-----------------------------------------------')

    def set_input(self, input, val_set):
        if not val_set:
            self.real_A = self.Tensor(self.batchSize, self.input_nc, self.cropSize[0], self.cropSize[1])
            self.real_B = self.Tensor(self.batchSize, self.input_nc, self.cropSize[0], self.cropSize[1])
            input_A = input['A']
            self.real_A_path = input['path']
            self.real_A.resize_(input_A.size()).copy_(input_A)
            self.DA     = input['DA'][0]
            if self.isTrain:
                input_B    = input['B']
                self.real_B.resize_(input_B.size()).copy_(input_B)
                self.DB     = input['DB'][0]
        else:
            raise NotImplementedError
            # self.real_A        = self.Tensor(self.batchSize, self.input_nc,
            #                                  self.loadSize[0], self.loadSize[1])
            # self.real_B        = self.Tensor(self.batchSize, self.input_nc,
            #                                  self.loadSize[0], self.loadSize[1])
            # self.real_A_FlowGT = self.Tensor(self.batchSize, self.input_nc,
            #                                  self.loadSize[0], self.loadSize[1])
            # input_A            = input['A']
            # input_A_FlowGT     = input['A_GT']
            # self.real_A.resize_(input_A.size()).copy_(input_A)
            # self.real_A_FlowGT.resize_(input_A_FlowGT.size()).copy_(input_A_FlowGT)
            # self.DA = input['DA'][0]
            # self.DB = input['DB'][0]
        self.image_paths = input['path']

    # To modify the test code later (add Flow predictions)- AA, 09/10/18, 11:08am
    def test(self):
        raise NotImplementedError

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, pred_real, fake, domain):
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D = self.criterionGAN(pred_real, pred_fake, True) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #D_A
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.pred_real_B, fake_B, self.DB)
        #D_B
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.pred_real_A, fake_A, self.DA)

    def backward_G(self):
        encoded_A_left    = self.netG.encode(self.real_A[:, 0:self.input_nc, :, :], self.DA)
        encoded_A_right   = self.netG.encode(self.real_A[:, self.input_nc:2*self.input_nc, :, :],
                                             self.DA)
        encoded_B_left    = self.netG.encode(self.real_B[:, 0:self.input_nc, :, :], self.DB)
        encoded_B_right   = self.netG.encode(self.real_B[:, self.input_nc:2*self.input_nc, :, :],
                                             self.DB)
        # GAN loss
        # D_A(G_A(A))
        self.fake_B  = torch.cat((self.netG.decode(encoded_A_left,  self.DB), \
                                  self.netG.decode(encoded_A_right, self.DB)), 1)
        pred_fake    = self.netD.forward(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(self.pred_real_B, pred_fake, False)
        # D_B(G_B(B))
        self.fake_A  = torch.cat((self.netG.decode(encoded_B_left,  self.DA), \
                                  self.netG.decode(encoded_B_right, self.DA)), 1)
        pred_fake    = self.netD.forward(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(self.pred_real_A, pred_fake, False)
        # Cycle losses
        # Forward cycle loss
        rec_encoded_A_left  = self.netG.encode(self.fake_B[:, 0:self.input_nc, :, :], self.DB)
        rec_encoded_A_right = self.netG.encode(self.fake_B[:, self.input_nc:2*self.input_nc, :, :],
                                               self.DB)
        self.rec_A          = torch.cat((self.netG.decode(rec_encoded_A_left,  self.DA), \
                                         self.netG.decode(rec_encoded_A_right, self.DA)), 1)
        self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A)
        # Backward cycle loss
        rec_encoded_B_left  = self.netG.encode(self.fake_A[:, 0:self.input_nc, :, :], self.DA)
        rec_encoded_B_right = self.netG.encode(self.fake_A[:, self.input_nc:2*self.input_nc, :, :],
                                               self.DA)
        self.rec_B          = torch.cat((self.netG.decode(rec_encoded_B_left,  self.DB), \
                                         self.netG.decode(rec_encoded_B_right, self.DB)), 1)
        self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B)

        # combined loss
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) * self.lambda_cyc
        loss_G.backward(retain_graph=True)

    def backward_netFlow(self):
        #######################################################################################
        # self.netFlow.param_reguires_grad(self.DB, False) # Gradients not needed for FlowNet_B
        # self.netFlow.param_reguires_grad(self.DA, True)  # Gradients needed for FlowNet_A
        # Forward pass along the netFlow_B predicting output for the real daytime images (real_B)
        flow1_real_B, flow2_real_B, \
        flow3_real_B, flow4_real_B, \
        flow5_real_B = self.netFlow.forward(self.real_B[:, 0:self.input_nc, :, :], \
                                            self.real_B[:, self.input_nc:2*self.input_nc, :, :],
                                            self.DB)
        # Forward pass along the netFlow_A predicting output for the fake nighttime images (fake_A)
        flow1_fake_A,  flow2_fake_A, \
        flow3_fake_A , flow4_fake_A, \
        flow5_fake_A = self.netFlow.forward(self.fake_A[:, 0:self.input_nc, :, :],
                                            self.fake_A[:, self.input_nc:2*self.input_nc, :, :],
                                            self.DA)
        # print(torch.min(self.fake_A[:, 0:self.input_nc, :, :]),
        #       torch.max(self.fake_A[:, 0:self.input_nc, :, :]))
        flow_fake_A = [flow1_fake_A, flow2_fake_A, flow3_fake_A, flow4_fake_A, flow5_fake_A]
        # Compute the EPE loss between the predictions
        self.loss_flowrec[self.DA] = multiscaleEPE(flow_fake_A, flow1_real_B.detach()) * \
                                     self.lambda_flowrec
        # Back-propagate the loss
        loss_netFlow_1 = self.loss_flowrec[self.DA]
        loss_netFlow_1.backward()
        ##########################################################################################
        # # self.netFlow.param_reguires_grad(self.DB, False) # Gradients not needed for FlowNet_B
        # # self.netFlow.param_reguires_grad(self.DA, False) # Gradients not needed for FlowNet_A
        # Forward pass along the netFlow_A predicting output for the rec. nighttime images (rec_A)
        flow1_rec_A, flow2_rec_A, \
        flow3_rec_A, flow4_rec_A, \
        flow5_rec_A = self.netFlow.forward(self.rec_A[:, 0:self.input_nc, :, :], \
                                           self.rec_A[:, self.input_nc:2*self.input_nc, :, :],
                                           self.DA)
        # Forward pass along the netFlow_B)predicting output for the fake daytime images (fake_B)
        flow1_fake_B, flow2_fake_B, \
        flow3_fake_B, flow4_fake_B, \
        flow5_fake_B = self.netFlow.forward(self.fake_B[:, 0:self.input_nc, :, :],\
                                            self.fake_B[:, self.input_nc:2*self.input_nc, :, :],
                                            self.DB)
        flow_fake_B = [flow1_fake_B, flow2_fake_B, flow3_fake_B, flow4_fake_B, flow5_fake_B]
        # Compute the EPE loss between the predictions
        self.loss_flowrec[self.DB] = multiscaleEPE(flow_fake_B, flow1_rec_A.detach()) * \
                                     self.lambda_flowrec
        # self.loss_flowrec[self.DB] = 0.0
        # Back-propagate the loss
        loss_netFlow_2 = self.loss_flowrec[self.DB]
        loss_netFlow_2.backward()
        #########################################################################################

        # Store data for visuals (only the first image of the mini-batch is sufficient)
        self.flow_rec_A  = flow1_rec_A[0, :, :].detach().cpu().numpy().transpose((1, 2, 0))
        self.flow_real_B = flow1_real_B[0, :, :].detach().cpu().numpy().transpose((1, 2, 0))
        self.flow_fake_A = flow1_fake_A[0, :, :].detach().cpu().numpy().transpose((1, 2, 0))
        self.flow_fake_B = flow1_fake_B[0, :, :].detach().cpu().numpy().transpose((1, 2, 0))


    def optimize_parameters(self, epoch):
        # Set the netGs, netDs in train mode
        # Set the FlowNets in train (FlowNet_A) and eval FlowNet_B) mode respectively
        self.netG.net_in_trainmode(self.DB, True)      # Put netG_B in train() mode
        self.netG.net_in_trainmode(self.DA, True)      # Put netG_A in train() mode
        self.netD.net_in_trainmode(self.DB, True)      # Put netD_B in train() mode
        self.netD.net_in_trainmode(self.DA, True)      # Put netD_A in train() mode
        self.netFlow.net_in_trainmode(self.DB, False)  # Put FlowNet_B in eval() mode
        self.netFlow.net_in_trainmode(self.DA, True)   # Put FlowNet_A in train() mode
        # Forward predictions for the real_A/B images from their corresponding discriminators
        self.pred_real_A = self.netD.forward(self.real_A, self.DA)
        self.pred_real_B = self.netD.forward(self.real_B, self.DB)
        # [G_A and G_B] and  FlowNet_A (since FlowNet_B is always frozen)
        self.netG.zero_grads(self.DA, self.DB)
        self.netFlow.zero_grads(self.DA)
        self.backward_G()
        self.backward_netFlow()
        self.netG.step_grads(self.DA, self.DB)
        self.netFlow.step_grads(self.DA)
        # D_A and D_B
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)

    def perform_validation(self, epoch, epoch_iter, savedir1):
        raise NotImplementedError

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, cyc_losses, flowrec_losses = extract(self.loss_D), \
                                                         extract(self.loss_G), \
                                                         extract(self.loss_cycle),\
                                                         extract(self.loss_flowrec)
        # Modifying the code - AA, 02/10/18, 12:20pm
        errors_ret = OrderedDict()
        for i in range(len(D_losses)):
            errors_ret['D_'+str(i)] = D_losses[i]
        for i in range(len(G_losses)):
            errors_ret['G_'+str(i)] = G_losses[i]
        for i in range(len(cyc_losses)):
            errors_ret['Cyc_'+str(i)] = cyc_losses[i]
        for i in range(len(flowrec_losses)):
            errors_ret['FlowNet_'+str(i)] = flowrec_losses[i]
        return errors_ret

    def save_current_visuals(self, epoch, epoch_iter, savedir1, savedir2):
        # Visuals1
        if self.use_grayscale_images:
            inv_norm_  = util.inv_normalize_grayscale()
        else:
            inv_norm_  = util.inv_normalize()
        totensor_   = vtransforms.ToTensor()
        real_A_imgs = inv_norm_(torch.cat((self.real_A[0, self.input_nc:2*self.input_nc, :, :],
                                           self.real_A[0, 0:self.input_nc, :, :]),2).detach().cpu())
        fake_B_imgs = inv_norm_(torch.cat((self.fake_B[0, self.input_nc:2*self.input_nc, :, :],
                                           self.fake_B[0, 0:self.input_nc, :, :]),2).detach().cpu())
        rec_A_imgs  = inv_norm_(torch.cat((self.rec_A[0,  self.input_nc:2*self.input_nc, :, :],
                                           self.rec_A[0,  0:self.input_nc, :, :]),2).detach().cpu())
        real_B_imgs = inv_norm_(torch.cat((self.real_B[0, self.input_nc:2*self.input_nc, :, :],
                                           self.real_B[0, 0:self.input_nc, :, :]),2).detach().cpu())
        fake_A_imgs = inv_norm_(torch.cat((self.fake_A[0, self.input_nc:2*self.input_nc, :, :],
                                           self.fake_A[0, 0:self.input_nc, :, :]),2).detach().cpu())
        rec_B_imgs  = inv_norm_(torch.cat((self.rec_B[0,  self.input_nc:2*self.input_nc, :, :],
                                           self.rec_B[0,  0:self.input_nc, :, :]),2).detach().cpu())
        if self.use_grayscale_images:
            real_A_imgs = torch.cat((real_A_imgs, real_A_imgs, real_A_imgs), 0)
            fake_B_imgs = torch.cat((fake_B_imgs, fake_B_imgs, fake_B_imgs), 0)
            rec_A_imgs  = torch.cat((rec_A_imgs, rec_A_imgs, rec_A_imgs), 0)
            real_B_imgs = torch.cat((real_B_imgs, real_B_imgs, real_B_imgs), 0)
            fake_A_imgs = torch.cat((fake_A_imgs, fake_A_imgs, fake_A_imgs), 0)
            rec_B_imgs  = torch.cat((rec_B_imgs, rec_B_imgs, rec_B_imgs), 0)
        _, h, w2 = real_A_imgs.size()
        w        = w2/2
        flow_rec_A = resize(flow_utils.flowToImg(self.flow_rec_A).astype(np.float32)/255.0,
                            [h, w], mode='reflect').astype(np.float32)
        flow_rec_A  = totensor_(Image.fromarray(np.uint8(flow_rec_A*255.0)))
        flow_fake_B = resize(flow_utils.flowToImg(self.flow_fake_B).astype(np.float32) / 255.0,
                             [h, w], mode='reflect').astype(np.float32)
        flow_fake_B = totensor_(Image.fromarray(np.uint8(flow_fake_B * 255.0)))
        flow_real_B = resize(flow_utils.flowToImg(self.flow_real_B).astype(np.float32) / 255.0,
                             [h, w], mode='reflect').astype(np.float32)
        flow_real_B = totensor_(Image.fromarray(np.uint8(flow_real_B * 255.0)))
        flow_fake_A = resize(flow_utils.flowToImg(self.flow_fake_A).astype(np.float32) / 255.0,
                             [h, w], mode='reflect').astype(np.float32)
        flow_fake_A = totensor_(Image.fromarray(np.uint8(flow_fake_A * 255.0)))
        output_img1 = torch.cat((torch.cat((rec_A_imgs,  flow_rec_A,
                                            fake_B_imgs, flow_fake_B, real_A_imgs), 2),  \
                                 torch.cat((real_B_imgs, flow_real_B,
                                            fake_A_imgs, flow_fake_A, rec_B_imgs ), 2)), 1)
        savename = savedir1 + '/visual' + '_epoch' + str(epoch) + '_' + str(epoch_iter) + '.jpg'
        vutils.save_image(output_img1, savename)
        # Visuals2
        if savedir2 is not None:
            raise NotImplementedError


    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            # print(self.real_A_path)
            message += '%s: %.3f ' % (k, v)
        print(message)

    def plot_current_errors(self, exp_name, glb_iter, errors, logger):
        logger.add_scalars(exp_name + '/G', {'G_0':errors['G_0'], 'G_1':errors['G_1']}, glb_iter)
        logger.add_scalars(exp_name + '/D', {'D_0': errors['D_0'], 'D_1': errors['D_1']}, glb_iter)
        logger.add_scalars(exp_name + '/Cyc', {'Cyc_0': errors['Cyc_0'],
                                               'Cyc_1': errors['Cyc_1']}, glb_iter)
        logger.add_scalars(exp_name + '/FlowNet', {'FlowNet_0': errors['FlowNet_0'],
                                                   'FlowNet_1': errors['FlowNet_1']}, glb_iter)


    def save(self, label):
        self.save_network(self.netG,    'G',       label, self.gpu_ids)
        self.save_network(self.netFlow, 'FlowNet', label, self.gpu_ids)
        self.save_network(self.netD,    'D',       label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netFlow.update_lr(new_lr/10)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)
