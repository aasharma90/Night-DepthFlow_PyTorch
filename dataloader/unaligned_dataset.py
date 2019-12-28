import os.path, glob
import torchvision.transforms as transforms
from dataloader.base_dataset import BaseDataset, perform_paired_transform
from dataloader.image_folder import make_dataset, make_dataset_from_txt
from PIL import Image
import random
import torch
import numpy as np

class UnalignedDataset(BaseDataset):
    def __init__(self, opt, dataname, val_set=False):
        super(UnalignedDataset, self).__init__()
        self.opt     = opt
        self.val_set = val_set
        self.dataname= dataname
        if not isinstance (opt.loadSize, (list,)):
            self.opt.loadSize = self.opt.loadSize.strip('[]').split(', ')
            self.opt.loadSize = [int(item) for item in self.opt.loadSize]
        if not isinstance (opt.cropSize, (list, )):
            self.opt.cropSize = self.opt.cropSize.strip('[]').split(', ')
            self.opt.cropSize = [int(item) for item in self.opt.cropSize]

        if opt.isTrain:
            if not self.val_set:
                self.data_files = [opt.trn_datafile1, opt.trn_datafile2]
                if self.dataname == 'Oxford':
                    self.paths_left = [make_dataset_from_txt(f, match_substr='left') for f in self.data_files]
                    self.paths_right= [make_dataset_from_txt(f, match_substr='right') for f in self.data_files]
                else:
                    print('No other dataset supported yet!')
                    raise NotImplementedError()
            else:
                self.data_files = [opt.val_datafile1]
                if self.dataname == 'Oxford':
                    self.paths_left = [sorted(make_dataset_from_txt(f, match_substr='left')) for f in self.data_files]
                    self.paths_right= [sorted(make_dataset_from_txt(f, match_substr='right')) for f in self.data_files]
                    self.paths_disp = [sorted(make_dataset_from_txt(f, match_substr='disp_GT')) for f in self.data_files]
                else:
                    print('No other dataset supported yet!')
                    raise NotImplementedError()
        else:
            raise NotImplementedError()
        self.sizes  = [len(p) for p in self.paths_left]

    def load_image(self, dom, idx, crop_imgs):
        # Read the left image
        path_left = self.paths_left[dom][idx]
        img_left  = Image.open(path_left).convert('RGB')
        # Read the right image
        path_right= self.paths_right[dom][idx]
        img_right = Image.open(path_right).convert('RGB')
        # Transform (pre-process) the two images
        img_left, img_right = perform_paired_transform(img_left, img_right,
                                                       crop_imgs, self.opt)
        # Concatenate the two images
        img       = torch.cat((img_left, img_right), 0)
        path      = [path_left, path_right]
        return img, path

    def load_disp(self, dom, idx):
        # Read the sparse disp GT of the left image
        path_disp      = self.paths_disp[dom][idx]
        disp_left      = Image.open(path_disp)
        w_orig, h_orig = disp_left.size
        # Resize the disp image to (opt.fineSize[0], opt.fineSize[1])
        nearest_resize_ = transforms.Resize((self.opt.loadSize[0], self.opt.loadSize[1]), Image.NEAREST)
        disp_left       = nearest_resize_(disp_left)
        w_new, h_new    = disp_left.size
        # Convert the image to a numpy array
        disp_left       = np.ascontiguousarray(disp_left, dtype=np.float32)/256.0
        # Perform the desired scaling (since the image has been re-sized)
        disp_left       = disp_left/(w_orig/w_new)
        return disp_left

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
            A_img, A_path = self.load_image(DA, index_A, crop_imgs=False)
            A_img_dispGT  = self.load_disp(DA, index_A)
            bundle = {'A': A_img, 'A_GT':A_img_dispGT, 'DA': DA, 'path': A_path, 'DB' : DB}

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
