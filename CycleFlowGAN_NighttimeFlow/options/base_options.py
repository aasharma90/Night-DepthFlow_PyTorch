import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', required=True, type=str, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataname', required=True, type=str, help='name of the dataset to check the model on?')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./ckpts/', help='models are saved here')

        # Changed to args to accept two text files containing the images - AA, 02/10/18, 10:05am
        self.parser.add_argument('--trn_datafile1', required=True, type=str, help='list containing trn images from dom1')
        self.parser.add_argument('--trn_datafile2', required=True, type=str, help='list containing trn images from dom2')

        self.parser.add_argument('--val_datafile1', required=True, type=str,help='list containing val images from dom1')

        # Fixed the n_domains to 2 by default (default CycleGAN) - AA, 02/10/18, 10:05am
        self.parser.add_argument('--n_domains', type=int, default=2, help='Number of domains to transfer among')

        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize|resize_and_crop|crop]')

        # Changing the format of loadSize and fineSize from int to str - AA, 02/10/18, 11:24am
        self.parser.add_argument('--loadSize', type=str, default="[256, 512]", help='scale images to this size')
        self.parser.add_argument('--cropSize', type=str, default="[256, 512]", help='then crop to this size')

        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--use_grayscale_images', action='store_true', help='if specified, use grayscale images (option overrides input/output_nc to 1')
        # Changing the number of input and output channels to 6 (for inputting and outputting stereo pair images)  - AA, 04/10/18, 5:37pm
        # Changing them back to 3 - AA, 19/10/18, 1:42pm
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        # Changing the number of filters to 32 from 64 by default - AA, 19/10/18, 1:42pm
        self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--netG_n_blocks', type=int, default=9, help='number of residual blocks to use for netG')
        self.parser.add_argument('--netG_n_shared', type=int, default=0, help='number of blocks to use for netG shared center module')
        self.parser.add_argument('--netD_n_layers', type=int, default=4, help='number of layers to use for netD')

        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='insert dropout for the generator')

        # Making it multi-gpu by default - AA, 02/10/18, 10:20am
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # Changing the default value to 8 - AA, 02/10/18, 10:30am
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir    = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        logs_dir    = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'logs')
        visuals1_dir= os.path.join(self.opt.checkpoints_dir, self.opt.name, 'visuals1')
        visuals2_dir= os.path.join(self.opt.checkpoints_dir, self.opt.name, 'visuals2')
        val_res_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'val_checks')
        nets_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'nets')
        util.mkdirs(logs_dir)
        util.mkdirs(visuals1_dir)
        util.mkdirs(visuals2_dir)
        util.mkdirs(val_res_dir)
        util.mkdirs(nets_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
