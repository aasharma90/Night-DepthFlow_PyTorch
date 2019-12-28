import torch.utils.data as data
from PIL import Image
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_func
from skimage.util import random_noise

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        transform_list.append(transforms.Resize((opt.loadSize[0], opt.loadSize[1]), Image.BICUBIC))

    if opt.isTrain:
        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.RandomCrop((opt.cropSize[0], opt.cropSize[1])))


    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

def perform_paired_transform(left_img, right_img, crop_imgs, opt):

    to_tensor_ = transforms.ToTensor()
    to_PIL_    = transforms.ToPILImage()
    if 'resize' in opt.resize_or_crop:
        resize_ = transforms.Resize((opt.loadSize[0], opt.loadSize[1]), Image.BICUBIC)
        left_img    = resize_(left_img)
        right_img   = resize_(right_img)
    if opt.isTrain:
        if 'crop' in opt.resize_or_crop and crop_imgs:
            w, h   = left_img.size
            th, tw = opt.cropSize[0], opt.cropSize[1]
            x1     = random.randint(0, w - tw)
            y1     = random.randint(0, h - th)
            left_img   = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img  = right_img.crop((x1, y1, x1 + tw, y1 + th))

    normalize_ = transforms.Normalize((0.5, 0.5, 0.5),
                                      (0.5, 0.5, 0.5))
    left_img   = normalize_(to_tensor_(left_img))
    right_img  = normalize_(to_tensor_(right_img))
    return left_img, right_img