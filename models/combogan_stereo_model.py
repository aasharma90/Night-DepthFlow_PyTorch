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
# import matlab.engine

class ComboGANstereoModel(BaseModel):
    def name(self):
        return 'ComboGANstereoModel'

    def __init__(self, opt):
        super(ComboGANstereoModel, self).__init__(opt)

        # Start the matlab engine
        # self.matlab_eng   = matlab.engine.start_matlab()

        self.n_domains    = opt.n_domains
        self.maxdisp      = opt.maxdisp
        self.model_dispnet= opt.model_dispnet
        self.DA, self.DB  = None, None
        self.loadSize     = opt.loadSize
        self.cropSize     = opt.cropSize
        self.batchSize    = opt.batchSize
        self.input_nc     = opt.input_nc

        # load/define networks
        self.netG   = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout_netG, 
                                      self.gpu_ids)
        self.netDisp = networks.define_DispNet(opt.maxdisp, opt.model_dispnet,
                                               False, self.n_domains, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, self.Tensor, opt.norm, 
                                          self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            self.netDisp.load(os.path.join(self.save_dir, '%d_net_%s' % (which_epoch, 'DispNet0.pth')), \
                              os.path.join(self.save_dir, '%d_net_%s' % (which_epoch, 'DispNet1.pth')))
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)
        else:
            self.netDisp.load(opt.loadmodel_dispnet, opt.loadmodel_dispnet)

        if self.isTrain:
            self.fake_pools     = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            self.L1             = torch.nn.SmoothL1Loss()
            self.downsample     = torch.nn.AvgPool2d(3, stride=2)
            self.criterionCycle = self.L1
            self.criterionGAN   = lambda r,f,v : (networks.GANLoss(r[0], f[0], v) + \
                                                  networks.GANLoss(r[1], f[1], v) + \
                                                  networks.GANLoss(r[2], f[2], v) + \
                                                  networks.GANLoss(r[3], f[3], v) + \
                                                  networks.GANLoss(r[4], f[4], v) + \
                                                  networks.GANLoss(r[5], f[5], v)) / 6

            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam,    opt.lr, (opt.beta1, 0.999))
            self.netDisp.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam,    opt.lr, (opt.beta1, 0.999))
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle   = [0]*self.n_domains
            self.loss_disprec = [0]*self.n_domains
            self.loss_FrFtsrec= [0]*self.n_domains
            # initialize loss multipliers
            self.lambda_cyc     = opt.lambda_cycle
            self.lambda_disprec = opt.lambda_disprec

        print('---------- Networks initialized ---------------')
        print('-----------------------------------------------')


    def set_input(self, input, val_set):
        if not val_set:
            self.real_A = self.Tensor(self.batchSize, self.input_nc, self.cropSize[0], self.cropSize[1])
            self.real_B = self.Tensor(self.batchSize, self.input_nc, self.cropSize[0], self.cropSize[1])
            input_A = input['A']
            self.real_A.resize_(input_A.size()).copy_(input_A)
            self.DA = input['DA'][0]
            if self.isTrain:
                input_B    = input['B']
                self.real_B.resize_(input_B.size()).copy_(input_B)
                self.DB = input['DB'][0]
        else:
            self.real_A        = self.Tensor(self.batchSize, self.input_nc,
                                             self.loadSize[0], self.loadSize[1])
            self.real_B        = self.Tensor(self.batchSize, self.input_nc,
                                             self.loadSize[0], self.loadSize[1])
            self.real_A_dispGT = self.Tensor(self.batchSize, self.input_nc,
                                             self.loadSize[0], self.loadSize[1])
            input_A            = input['A']
            input_A_dispGT     = input['A_GT']
            self.real_A.resize_(input_A.size()).copy_(input_A)
            self.real_A_dispGT.resize_(input_A_dispGT.size()).copy_(input_A_dispGT)
            self.DA = input['DA'][0]
            self.DB = input['DB'][0]
        self.image_paths = input['path']

    
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

        # Combined loss
        loss_G = (self.loss_G[self.DA] + self.loss_G[self.DB]) + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) * self.lambda_cyc 
        loss_G.backward(retain_graph=True)


    def backward_netDisp(self):
        if self.model_dispnet == 'PSMNet_stackhourglass':
            ################################################################################
            # self.netDisp.param_reguires_grad(self.DB, False) # Gradients not needed for DispNet_B
            # self.netDisp.param_reguires_grad(self.DA, True)  # Gradients needed for DispNet_A
            # Forward pass along the DispNet (netDisp_B)  predicting for real daytime images (real_B)
            disp1_real_B, \
            disp2_real_B, \
            disp3_real_B = self.netDisp.forward(self.real_B[:, 0:self.input_nc, :, :], \
                                                self.real_B[:, self.input_nc:2*self.input_nc, :, :],
                                                self.DB)
            # Forward pass along the DispNet (netDisp_A) predicting for fake nighttime images (fake_A)
            disp1_fake_A, \
            disp2_fake_A, \
            disp3_fake_A = self.netDisp.forward(self.fake_A[:, 0:self.input_nc, :, :], \
                                                self.fake_A[:, self.input_nc:2*self.input_nc, :, :],
                                                self.DA)
            # Compute the L1 loss b/w the predictions
            self.loss_disprec[self.DA] = self.L1(disp1_fake_A, disp1_real_B.detach()) + \
                                         self.L1(disp2_fake_A, disp2_real_B.detach()) + \
                                         self.L1(disp3_fake_A, disp3_real_B.detach())
            # Back-propagate the loss
            loss_netDisp_1 = self.loss_disprec[self.DA] * self.lambda_disprec
            loss_netDisp_1.backward()
            ###################################################################################
            # self.netDisp.param_reguires_grad(self.DB, False) # Gradients not needed for DispNet_B
            # self.netDisp.param_reguires_grad(self.DA, False) # Gradients not needed for DispNet_A
            # Forward pass along the DispNet (netDisp_A) predicting for rec. nighttime images (rec_A)
            disp1_rec_A,  \
            disp2_rec_A,  \
            disp3_rec_A  = self.netDisp.forward(self.rec_A[:, 0:self.input_nc, :, :], \
                                                self.rec_A[:, self.input_nc:2*self.input_nc, :, :],
                                                self.DA)
            # Forward pass along the DispNet (netDisp_B) predicting for fake daytime images (fake_B)
            disp1_fake_B, \
            disp2_fake_B, \
            disp3_fake_B = self.netDisp.forward(self.fake_B[:, 0:self.input_nc, :, :], \
                                                self.fake_B[:, self.input_nc:2*self.input_nc, :, :],
                                                self.DB)
            # Compute the L1 loss b/w the predictions
            self.loss_disprec[self.DB] = self.L1(disp1_fake_B, disp1_rec_A.detach()) + \
                                         self.L1(disp2_fake_B, disp2_rec_A.detach()) + \
                                         self.L1(disp3_fake_B, disp3_rec_A.detach())
            # Back-propagate the loss
            loss_netDisp_2 = self.loss_disprec[self.DB] * self.lambda_disprec
            loss_netDisp_2.backward()
            #######################################################################################
        elif self.model_dispnet == 'PSMNet_basic':
            #######################################################################################
            # self.netDisp.param_reguires_grad(self.DB, False) # Gradients not needed for DispNet_B
            # self.netDisp.param_reguires_grad(self.DA, True)  # Gradients needed for DispNet_A
            # Forward pass along the DispNet (netDisp_B)  predicting for real daytime images (real_B)
            disp3_real_B = self.netDisp.forward(self.real_B[:, 0:self.input_nc, :, :],
                                                self.real_B[:, self.input_nc:2*self.input_nc, :, :],
                                                self.DB)
            # Forward pass along the DispNet (netDisp_A) predicting for fake nighttime images (fake_A)
            disp3_fake_A = self.netDisp.forward(self.fake_A[:, 0:self.input_nc, :, :],
                                                self.fake_A[:, self.input_nc:2*self.input_nc, :, :],
                                                self.DA)
            # Compute the L1 loss b/w the predictions
            self.loss_disprec[self.DA] = self.L1(disp3_fake_A, disp3_real_B.detach())
            # Back-propagate the loss
            loss_netDisp_1 = self.loss_disprec[self.DA] * self.lambda_disprec
            loss_netDisp_1.backward()
            #######################################################################################
            # self.netDisp.param_reguires_grad(self.DB, False) # Gradients not needed for DispNet_B
            # self.netDisp.param_reguires_grad(self.DA, False) # Gradients not needed for DispNet_A
            # Forward pass along the DispNet (netDisp_A) predicting for rec. nighttime images (rec_A)
            disp3_rec_A  = self.netDisp.forward(self.rec_A[:, 0:self.input_nc, :, :],
                                                self.rec_A[:, self.input_nc:2*self.input_nc, :, :],
                                                self.DA)
            # Forward pass along the DispNet (netDisp_B) predicting for fake daytime images (fake_B)
            disp3_fake_B = self.netDisp.forward(self.fake_B[:, 0:self.input_nc, :, :],
                                                self.fake_B[:, self.input_nc:2*self.input_nc, :, :],
                                                self.DB)
            # Compute the L1 loss b/w the predictions
            self.loss_disprec[self.DB] =  self.L1(disp3_fake_B, disp3_rec_A.detach())
            # Back-propagate the loss
            loss_netDisp_2 = self.loss_disprec[self.DB] * self.lambda_disprec
            loss_netDisp_2.backward()
            ########################################################################################

        # Store data for visuals (only the first image of the mini-batch is sufficient)
        self.disp_rec_A  = disp3_rec_A[0, :, :].detach().cpu().numpy()/self.maxdisp
        self.disp_real_B = disp3_real_B[0, :, :].detach().cpu().numpy()/self.maxdisp
        self.disp_fake_A = disp3_fake_A[0, :, :].detach().cpu().numpy()/self.maxdisp
        self.disp_fake_B = disp3_fake_B[0, :, :].detach().cpu().numpy()/self.maxdisp


    def optimize_parameters(self, epoch):
        # Set the netGs, netDs in train mode
        # Set the DispNets in train (DispNet_A) and eval DispNet_B) mode respectively
        self.netG.net_in_trainmode(self.DB, True)      # Put netG_B in train() mode
        self.netG.net_in_trainmode(self.DA, True)      # Put netG_A in train() mode
        self.netD.net_in_trainmode(self.DB, True)      # Put netD_B in train() mode
        self.netD.net_in_trainmode(self.DA, True)      # Put netD_A in train() mode
        self.netDisp.net_in_trainmode(self.DB, False)  # Put DispNet_B in eval() mode
        self.netDisp.net_in_trainmode(self.DA, True)   # Put DispNet_A in train() mode
        # Forward predictions for the real_A/B images from their corresponding discriminators
        self.pred_real_A = self.netD.forward(self.real_A, self.DA)
        self.pred_real_B = self.netD.forward(self.real_B, self.DB)
        # [G_A and G_B] and  DispNet_A (since DispNet_B is always frozen)
        self.netG.zero_grads(self.DA, self.DB)
        self.netDisp.zero_grads(self.DA)
        self.backward_G()
        self.backward_netDisp()
        self.netG.step_grads(self.DA, self.DB)
        self.netDisp.step_grads(self.DA)
        # D_A and D_B
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)

    def perform_validation(self, epoch, epoch_iter, savedir1):
        cm_jet_     = cm.get_cmap('jet')
        inv_norm_   = util.inv_normalize()
        totensor_   = vtransforms.ToTensor()
        self.netG.net_in_trainmode(self.DB, False)     # Put netG_B in eval() mode
        self.netG.net_in_trainmode(self.DA, False)     # Put netG_A in eval() mode
        self.netD.net_in_trainmode(self.DB, False)     # Put netD_B in eval() mode
        self.netD.net_in_trainmode(self.DA, False)     # Put netD_A in eval() mode
        self.netDisp.net_in_trainmode(self.DB, False)  # Put DispNet_B in eval() mode
        self.netDisp.net_in_trainmode(self.DA, False)  # Put DispNet_A in eval() mode
        with torch.no_grad():
            # Get disparity ground-truth of the image
            disp3_real_A_GT = self.real_A_dispGT
            if epoch == 0:
                _, _,  disp3_rec_A  = self.netDisp.forward(self.real_A[:, 0:self.input_nc, :, :], \
                                                           self.real_A[:, self.input_nc:2*self.input_nc, :, :], self.DA)
            else:
                # Get predictions from reconstructed images rec_A
                encoded_A_left      = self.netG.encode(self.real_A[:, 0:self.input_nc, :, :], self.DA)
                encoded_A_right     = self.netG.encode(self.real_A[:, self.input_nc:2*self.input_nc, :, :], self.DA)
                fake_B              = torch.cat((self.netG.decode(encoded_A_left,  self.DB), \
                                                 self.netG.decode(encoded_A_right, self.DB)), 1)
                rec_encoded_A_left  = self.netG.encode(fake_B[:, 0:self.input_nc, :, :], self.DB)
                rec_encoded_A_right = self.netG.encode(fake_B[:, self.input_nc:2*self.input_nc, :, :], self.DB)
                rec_A               = torch.cat((self.netG.decode(rec_encoded_A_left,  self.DA), \
                                                 self.netG.decode(rec_encoded_A_right, self.DA)), 1)
                _, _,  disp3_rec_A  = self.netDisp.forward(rec_A[:, 0:self.input_nc, :, :], \
                                                           rec_A[:, self.input_nc:2*self.input_nc, :, :], self.DA)
                rec_A_imgs  = inv_norm_(torch.cat((rec_A[0, self.input_nc:2*self.input_nc, :, :], rec_A[0, 0:self.input_nc, :, :]),2).detach().cpu())
                fake_B_imgs = inv_norm_(torch.cat((fake_B[0, self.input_nc:2*self.input_nc, :, :], fake_B[0, 0:self.input_nc, :, :]),2).detach().cpu())

            # Compute validation loss (3px error) (adopted from PSMNet's finetune.py code)
            disp_pred       = disp3_rec_A.data.cpu()
            # computing 3-px error#
            disp_true       = disp3_real_A_GT.data.cpu()
            # vutils.save_image(disp_pred[0, :, :]/self.maxdisp, 'pred.png')
            # vutils.save_image(disp_true[0, :, :]/self.maxdisp, 'true.png')
            true_disp       = disp_true
            index           = np.argwhere(true_disp > 0)
            disp_true[index[0][:], index[1][:], index[2][:]] \
                            = np.abs(true_disp[index[0][:], index[1][:], index[2][:]] -
                                     disp_pred[index[0][:], index[1][:], index[2][:]])
            correct         = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) + \
                              (disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
                               index[0][:], index[1][:], index[2][:]] * 0.05)
            validation_loss =  1 - (float(torch.sum(correct)) / float(len(index[0])))

            # Visuals
            real_A_imgs    = inv_norm_(torch.cat((self.real_A[0, self.input_nc:2*self.input_nc, :, :], self.real_A[0, 0:self.input_nc, :, :]),2).detach().cpu())
            disp_real_A_GT = disp3_real_A_GT[0, :, :].cpu()/self.maxdisp
            disp_real_A_GT = totensor_(Image.fromarray(np.uint8(cm_jet_(disp_real_A_GT)[:, :, :-1]*255.0)))
            disp_real_A_GT_1 = disp_real_A_GT[0, :, :]
            disp_real_A_GT_2 = disp_real_A_GT[1, :, :]
            disp_real_A_GT_3 = disp_real_A_GT[2, :, :]
            # print(disp_real_A_GT_1)
            # print(disp_real_A_GT_2)
            # print(disp_real_A_GT_3)
            # exit()
            true_tensor      = torch.ones_like(disp_real_A_GT_1 == 0.0)
            false_tensor     = torch.zeros_like(disp_real_A_GT_1 == 0.0)
            invalid_disp     = torch.where((disp_real_A_GT_1==0.0) &\
                                           (disp_real_A_GT_2==0.0) &\
                                           (disp_real_A_GT_3<0.4981), true_tensor, false_tensor)# ROUGH ESTIMATION!
            disp_real_A_GT_1[invalid_disp] = 0.0
            disp_real_A_GT_2[invalid_disp] = 0.0
            disp_real_A_GT_3[invalid_disp] = 0.0
            disp_real_A_GT[0, :, :]        = disp_real_A_GT_1
            disp_real_A_GT[1, :, :]        = disp_real_A_GT_2
            disp_real_A_GT[2, :, :]        = disp_real_A_GT_3
            disp_rec_A    = disp3_rec_A[0, :, :].detach().cpu().numpy()/self.maxdisp
            disp_rec_A    = totensor_(Image.fromarray(np.uint8(cm_jet_(disp_rec_A)[:, :, :-1]*255.0)))
            if epoch == 0:
                output_img1 = torch.cat((real_A_imgs, disp_real_A_GT, \
                                         torch.zeros_like(real_A_imgs),real_A_imgs, disp_rec_A), 2)
            else:
                output_img1 = torch.cat((real_A_imgs, disp_real_A_GT, \
                                         fake_B_imgs, rec_A_imgs, disp_rec_A), 2)
            vutils.save_image(output_img1, savedir1 + '/visual' +
                              '_epoch' + str(epoch) + '_' + str(epoch_iter) + '.jpg')
            message = '(epoch: %d, iters: %d, val_loss: %.3f) ' % (epoch, epoch_iter, validation_loss)
            print(message)
            return validation_loss

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, \
        cyc_losses, disprec_losses = extract(self.loss_D), extract(self.loss_G), \
                                     extract(self.loss_cycle), extract(self.loss_disprec)
        
        errors_ret = OrderedDict()
        for i in range(len(D_losses)):
            errors_ret['D_'+str(i)] = D_losses[i]
        for i in range(len(G_losses)):
            errors_ret['G_'+str(i)] = G_losses[i]
        for i in range(len(cyc_losses)):
            errors_ret['Cyc_'+str(i)] = cyc_losses[i]
        for i in range(len(disprec_losses)):
            errors_ret['DispNet_'+str(i)] = disprec_losses[i]
        return errors_ret


    def save_current_visuals(self, epoch, epoch_iter, savedir1, savedir2):
        # Visuals1
        cm_jet_     = cm.get_cmap('jet')
        inv_norm_   = util.inv_normalize()
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

        disp_rec_A  = totensor_(Image.fromarray(np.uint8(cm_jet_(self.disp_rec_A)[:, :, :-1]*255.0)))
        disp_fake_B = totensor_(Image.fromarray(np.uint8(cm_jet_(self.disp_fake_B)[:, :, :-1]*255.0)))
        disp_real_B = totensor_(Image.fromarray(np.uint8(cm_jet_(self.disp_real_B)[:, :, :-1]*255.0)))
        disp_fake_A = totensor_(Image.fromarray(np.uint8(cm_jet_(self.disp_fake_A)[:, :, :-1]*255.0)))
        output_img1 = torch.cat((torch.cat((rec_A_imgs,  disp_rec_A,
                                            fake_B_imgs, disp_fake_B, real_A_imgs), 2),  \
                                 torch.cat((real_B_imgs, disp_real_B,
                                            fake_A_imgs, disp_fake_A, rec_B_imgs ), 2)), 1)
        vutils.save_image(output_img1, savedir1 + '/visual' +
                          '_epoch' + str(epoch) + '_' + str(epoch_iter) + '.jpg')
        # Visuals2
        if savedir2 is not None:
            real_A_imgs = np.uint8(real_A_imgs.permute(1, 2, 0).numpy()*255)
            real_A_imgs = np.asarray(self.matlab_eng.boostLIME(matlab.uint8(real_A_imgs.tolist())),
                                     dtype='float32')/255
            real_A_imgs = torch.from_numpy(real_A_imgs).permute(2, 0, 1)
            fake_A_imgs = np.uint8(fake_A_imgs.permute(1, 2, 0).numpy()*255)
            fake_A_imgs = np.asarray(self.matlab_eng.boostLIME(matlab.uint8(fake_A_imgs.tolist())),
                                     dtype='float32')/255
            fake_A_imgs = torch.from_numpy(fake_A_imgs).permute(2, 0, 1)
            rec_A_imgs  = np.uint8(rec_A_imgs.permute(1, 2, 0).numpy()*255)
            rec_A_imgs  = np.asarray(self.matlab_eng.boostLIME(matlab.uint8(rec_A_imgs.tolist())),
                                     dtype='float32')/255
            rec_A_imgs  = torch.from_numpy(rec_A_imgs).permute(2, 0, 1)
            output_img2 = torch.cat((rec_A_imgs,  torch.zeros_like(disp_rec_A), \
                                     fake_A_imgs, torch.zeros_like(disp_rec_A), real_A_imgs), 2)
            vutils.save_image(output_img2, savedir2 + '/visual' +
                              '_epoch' + str(epoch) + '_' + str(epoch_iter) + '.jpg')


    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        print(message)


    def plot_current_errors(self, exp_name, glb_iter, errors, logger):
        logger.add_scalars(exp_name + '/G', {'G_0':errors['G_0'],
                                             'G_1':errors['G_1']}, glb_iter)
        logger.add_scalars(exp_name + '/D', {'D_0': errors['D_0'],
                                             'D_1': errors['D_1']}, glb_iter)
        logger.add_scalars(exp_name + '/Cyc', {'Cyc_0': errors['Cyc_0'],
                                               'Cyc_1': errors['Cyc_1']}, glb_iter)
        logger.add_scalars(exp_name + '/DispNet', {'DispNet_0': errors['DispNet_0'],
                                                   'DispNet_1': errors['DispNet_1']}, glb_iter)


    def save(self, label):
        self.save_network(self.netG,    'G',       label, self.gpu_ids)
        self.save_network(self.netDisp, 'DispNet', label, self.gpu_ids)
        self.save_network(self.netD,    'D',       label, self.gpu_ids)


    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netDisp.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)
