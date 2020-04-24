import os.path, glob
import torchvision.transforms as transforms
from dataloader.base_dataset import BaseDataset, perform_paired_transform
from dataloader.image_folder import make_dataset, make_dataset_from_txt
from PIL import Image
import torch
import numpy as np

class UnalignedDataset(BaseDataset):
    def __init__(self, opt, dataname, val_set=False):

        # Setting manual seed (if needed)
        torch.manual_seed(0)

        super(UnalignedDataset, self).__init__()
        self.opt     = opt
        self.val_set = val_set
        self.dataname= dataname
        if not isinstance (opt.loadSize, (list,)):
            # Adding a change here to map loadSize from a string to a list - AA, 02/10/18, 11:27am
            self.opt.loadSize = self.opt.loadSize.strip('[]').split(', ')
            self.opt.loadSize = [int(item) for item in self.opt.loadSize]
        if not isinstance (opt.cropSize, (list, )):
            # Adding a change here to map cropSize from a string to a list - AA, 02/10/18, 11:27am
            self.opt.cropSize = self.opt.cropSize.strip('[]').split(', ')
            self.opt.cropSize = [int(item) for item in self.opt.cropSize]

        if opt.isTrain:
            if not self.val_set:
                self.data_files = [opt.trn_datafile1, opt.trn_datafile2]
                if self.dataname == 'LocalData' or dataname == 'Oxford':
                    self.paths_im0 = [make_dataset_from_txt(f, 0) for f in self.data_files]
                    self.paths_im1 = [make_dataset_from_txt(f, 1) for f in self.data_files]
                else:
                    raise NotImplementedError
            else:
                self.data_files = [opt.val_datafile1]
                raise NotImplementedError
        else:
            raise NotImplementedError()
        self.sizes  = [len(p) for p in self.paths_im0]

    def load_image(self, dom, idx, crop_imgs):
        use_grayscale_images = self.opt.use_grayscale_images
        # Read the im0 image
        path_im0 = self.paths_im0[dom][idx]
        if use_grayscale_images:
            img_im0 = Image.open(path_im0).convert('L')
        else:
            img_im0 = Image.open(path_im0).convert('RGB')
        # Read the im1 image
        path_im1= self.paths_im1[dom][idx]
        if use_grayscale_images:
            img_im1 = Image.open(path_im1).convert('L')
        else:
            img_im1 = Image.open(path_im1).convert('RGB')
        # Transform (pre-process) the two images
        img_im0, img_im1 = perform_paired_transform(img_im0, img_im1,
                                                    crop_imgs, self.opt, normalize=True)
        # Concatenate the two images
        img       = torch.cat((img_im0, img_im1), 0)
        path      = [path_im0, path_im1]
        return img, path

    def __getitem__(self, index):
        if not self.opt.isTrain:
            raise NotImplementedError()
        else:
            DA, DB  = [0, 1]
            index_A = index
        if not self.val_set:
            A_img, A_path = self.load_image(DA, index_A, crop_imgs=True)
            bundle = {'A': A_img, 'DA': DA, 'path': A_path}
        else:
            raise NotImplementedError

        if self.opt.isTrain and not self.val_set:
            index_B = index
            B_img, _ = self.load_image(DB, index_B, crop_imgs=True)
            bundle.update( {'B': B_img, 'DB': DB} )

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return min(self.sizes)
        return self.sizes[0]

    def name(self):
        return 'UnalignedDataset'
