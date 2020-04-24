import torch
import torch.nn as nn
from torch.nn import init
import functools, itertools
import numpy as np
from util.util import gkern_2d
from collections import OrderedDict
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def define_G(input_nc, output_nc, ngf, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if n_blocks_shared > 0:
        n_blocks_shdec = n_blocks_shared // 2
        n_blocks_shenc = n_blocks_shared - n_blocks_shdec
        shenc_args = (n_domains, n_blocks_shenc) + dup_args
        shdec_args = (n_domains, n_blocks_shdec) + dup_args
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args, ResnetGenShared, shenc_args, shdec_args)
    else:
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG


def define_D(input_nc, ndf, netD_n_layers, n_domains, tensor, norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netD = D_Plexer(n_domains, NLayerDiscriminator, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD

def define_FlowNet(use_grayscale_images, n_domains, gpu_ids=[]):
    model_args      =(use_grayscale_images, gpu_ids)
    plex_netFlowNet = FlowNet_Plexer(n_domains, PWCNet_model_, model_args)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netFlowNet.cuda(gpu_ids[0])
    plex_netFlowNet.apply(weights_init)
    return plex_netFlowNet

##############################################################################
# Losses
##############################################################################
# Defines the GAN loss which uses the Relativistic LSGAN
def GANLoss(inputs_real, inputs_fake, is_discr):
    if is_discr:
        y = -1
    else:
        y = 1
        inputs_real = [i.detach() for i in inputs_real]
    loss = lambda r,f : torch.mean((r-f+y)**2)
    losses = [loss(r,f) for r,f in zip(inputs_real, inputs_fake)]
    multipliers = list(range(1, len(inputs_real)+1));  multipliers[-1] += 1
    losses = [m*l for m,l in zip(multipliers, losses)]
    return sum(losses) / (sum(multipliers) * len(losses))

##############################################################################
# Classes
##############################################################################
# A slight modification to parallelize the data before inputting to PWCNet
class PWCNet_model_(nn.Module):
    def __init__(self, use_grayscale_images, gpu_ids=[]):
        super(PWCNet_model_, self).__init__()
        from .PWCNet import pwc_dc_net as PWCNet
        self.module  = PWCNet(path=None)
        self.gpu_ids = gpu_ids

    def forward(self, im0, im1):
        if self.gpu_ids and isinstance(im0.data, torch.cuda.FloatTensor) and isinstance(im1.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.module, (im0, im1), self.gpu_ids)
        return self.module(im0, im1)

# Defines the generator that consists of Resnet blocks between a few downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)

class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids     = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter= tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())
        self.input_nc    = input_nc

        self.model_rgb  = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        # if self.input_nc == 1:
        #     # Re-adjust the Gaussian filter if the images are in grayscale
        #     self.blur_filter = self.blur_filter[0:1, :, :, :]
        #     blurred_im0  = torch.nn.functional.conv2d(input[:, 0:1, :, :],
        #                                               self.blur_filter, groups=1, padding=2)
        #     blurred_im1  = torch.nn.functional.conv2d(input[:, 1:2, :, :],
        #                                               self.blur_filter, groups=1, padding=2)
        #     gray_im0     = input[:, 0:1, :, :]
        #     gray_im1     = input[:, 1:2, :, :]
        # elif self.input_nc == 3:
        #     blurred_im0 = torch.nn.functional.conv2d(input[:, 0:3, :, :],
        #                                              self.blur_filter, groups=3, padding=2)
        #     blurred_im1 = torch.nn.functional.conv2d(input[:, 3:6, :, :],
        #                                              self.blur_filter, groups=3, padding=2)
        #     gray_im0    = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)
        #     gray_im1    = (.299*input[:,3,:,:] + .587*input[:,4,:,:] + .114*input[:,5,:,:]).unsqueeze_(1)
        #
        # gray_dsamp_im0 = nn.functional.conv2d(gray_im0, self.dsamp_filter, stride=2)
        # gray_dsamp_im1 = nn.functional.conv2d(gray_im1, self.dsamp_filter, stride=2)
        # dx_im0         = nn.functional.conv2d(gray_dsamp_im0, self.grad_filter)
        # dy_im0         = nn.functional.conv2d(gray_dsamp_im0, self.grad_filter.transpose(-2,-1))
        # dx_im1         = nn.functional.conv2d(gray_dsamp_im1, self.grad_filter)
        # dy_im1         = nn.functional.conv2d(gray_dsamp_im1, self.grad_filter.transpose(-2,-1))
        # gradient_im0   = torch.cat([dx_im0,dy_im0], 1)
        # gradient_im1   = torch.cat([dx_im1, dy_im1], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            # outs1_im0 = nn.parallel.data_parallel(self.model_rgb, blurred_im0, self.gpu_ids)
            # outs1_im1 = nn.parallel.data_parallel(self.model_rgb, blurred_im1, self.gpu_ids)
            # outs2_im0 = nn.parallel.data_parallel(self.model_gray, gray_im0, self.gpu_ids)
            # outs2_im1 = nn.parallel.data_parallel(self.model_gray, gray_im1, self.gpu_ids)
            # outs3_im0 = nn.parallel.data_parallel(self.model_grad, gradient_im0, self.gpu_ids)
            # outs3_im1 = nn.parallel.data_parallel(self.model_grad, gradient_im1, self.gpu_ids)
            outs1_im0 = nn.parallel.data_parallel(self.model_rgb, input[:, 0:3, :, :], self.gpu_ids)
            outs1_im1 = nn.parallel.data_parallel(self.model_rgb, input[:, 3:6, :, :], self.gpu_ids)
        else:
            # outs1_im0 = self.model_rgb(blurred_im0)
            # outs1_im1 = self.model_rgb(blurred_im1)
            # outs2_im0 = self.model_gray(gray_im0)
            # outs2_im1 = self.model_gray(gray_im1)
            # outs3_im0 = self.model_grad(gradient_im0)
            # outs3_im1 = self.model_grad(gradient_im1)
            outs1_im0 = self.model_rgb(input[:, 0:3, :, :])
            outs1_im1 = self.model_rgb(input[:, 3:6, :, :])
        # return outs1_im0, outs1_im1, outs2_im0, outs2_im1, outs3_im0, outs3_im1
        return outs1_im0, outs1_im1


class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        for net in self.networks:
            net = nn.DataParallel(net)
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) \
                           for net in self.networks]

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))

class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.encoders = [encoder(*enc_args) for _ in range(n_domains)]
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]

        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append( self.shared_encoder )
            self.decoders.append( self.shared_decoder )
        self.networks = self.encoders + self.decoders

    # def load_(self, save_dir, save_epoch):
    #     for i, net in enumerate(self.networks):
    #         if i == 0:
    #             net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_G0.pth'))
    #         elif i==1:
    #             net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_G1.pth'))
    #         elif i==2:
    #             net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_G2.pth'))
    #         elif i==3:
    #             net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_G3.pth'))

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        for enc, dec in zip(self.encoders, self.decoders):
            params = itertools.chain(enc.parameters(), dec.parameters())
            self.optimizers.append( opt(params, lr=lr, betas=betas) )

    def net_in_trainmode(self, dom, cond):
        if cond:
            self.encoders[dom].train()
            self.decoders[dom].train()
        else:
            self.encoders[dom].eval()
            self.decoders[dom].eval()

    def forward(self, input, in_domain, out_domain):
        encoded = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def encode(self, input, domain):
        output = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain)
        return output

    def decode(self, input, domain):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        return self.decoders[domain].forward(input)

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        if self.sharing:
            self.optimizers[-1].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        if self.sharing:
            self.optimizers[-1].step()
        self.optimizers[dom_b].step()

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) +'\n'+ repr(d) +'\n'+ \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) +'\n'+ \
            'Number of parameters per Encoder: %d' % e_params +'\n'+ \
            'Number of parameters per Deocder: %d' % d_params

class FlowNet_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(FlowNet_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def init_optimizers(self, opt, lr, betas):
        # Only create the optimizer for the dom_a (or nighttime FlowNet)
        self.optimizers = [opt(self.networks[0].parameters(), lr=lr, betas=betas)]

    def net_in_trainmode(self, dom, cond):
        if cond:
            self.networks[dom].train()
        else:
            self.networks[dom].eval()

    def param_reguires_grad(self, dom, cond):
        for p in self.networks[dom].parameters():
            p.requires_grad = cond

    def forward(self, im0, im1, domain):
        Flownet = self.networks[domain]
        return Flownet.forward(im0, im1)

    def zero_grads(self, dom):
        self.optimizers[dom].zero_grad()

    def step_grads(self, dom):
        self.optimizers[dom].step()

    def load(self, save_path_0, save_path_1, opt):
        for i, net in enumerate(self.networks):
            if i == 0:
                if not opt.continue_train:
                    dict_0 = torch.load(save_path_0)
                else:
                    dict_0 = torch.load(save_path_0)['state_dict']
                new_dict_0 = OrderedDict()
                for k, v in dict_0.items():
                    if not opt.continue_train:
                        name = 'module.' + k
                    else:
                        name = k
                    new_dict_0[name] = v
                net.load_state_dict(new_dict_0)
            elif i==1:
                if not opt.continue_train:
                    dict_1 = torch.load(save_path_1)
                else:
                    dict_1 = torch.load(save_path_1)['state_dict']
                new_dict_1 = OrderedDict()
                for k, v in dict_1.items():
                    if not opt.continue_train:
                        name = 'module.' + k
                    else:
                        name = k
                    new_dict_1[name] = v
                net.load_state_dict(new_dict_1)

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save({'state_dict': net.state_dict()}, filename)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d FlowNets' % len(self.networks) +'\n'+ \
            'Number of parameters per FlowNet: %d' % t_params

class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    # def load_(self, save_dir, save_epoch):
    #     for i, net in enumerate(self.networks):
    #         if i == 0:
    #             net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_D0.pth'))
    #         elif i==1:
    #             net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_D1.pth'))

    def net_in_trainmode(self, dom, cond):
        if cond:
            self.networks[dom].train()
        else:
            self.networks[dom].eval()

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Discriminator: %d' % t_params


class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x

class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;  continue
            predictions.append( output[:,-1,:,:] )
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]
        return predictions

