import torch.utils.data as data
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_func

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
        # Modifying the code here to accpet a rectangular size - AA, 02/10/18, 11:28am
        transform_list.append(transforms.Resize((opt.loadSize[0], opt.loadSize[1]), Image.BICUBIC))

    if opt.isTrain:
        if 'crop' in opt.resize_or_crop:
            # Modifying the code here to accpet a rectangular size - AA, 02/10/18, 11:28am
            transform_list.append(transforms.RandomCrop((opt.cropSize[0], opt.cropSize[1])))
        # if not opt.no_flip: # Flipping is not allowed for stereo images! - AA, 12/10/18, 7:08pm
        #     transform_list.append(transforms.RandomHorizontalFlip())

    if opt.use_grayscale_images:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize([0.5], [0.5])]
    else:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

def perform_paired_transform(im0, im1, crop_imgs, opt, normalize=True):
    if 'resize' in opt.resize_or_crop:
        resize_ = transforms.Resize((opt.loadSize[0], opt.loadSize[1]), Image.BICUBIC)
        im0   = resize_(im0)
        im1   = resize_(im1)
    if opt.isTrain:
        if 'crop' in opt.resize_or_crop and crop_imgs:
            w, h   = im0.size
            th, tw = opt.cropSize[0], opt.cropSize[1]
            x1     = random.randint(0, w - tw)
            y1     = random.randint(0, h - th)
            im0    = im0.crop((x1, y1, x1 + tw, y1 + th))
            im1    = im1.crop((x1, y1, x1 + tw, y1 + th))

    to_tensor_ = transforms.ToTensor()
    if opt.use_grayscale_images:
        if normalize:
            normalize_ = transforms.Normalize([0.5], [0.5])
            im0 = normalize_(to_tensor_(im0))
            im1 = normalize_(to_tensor_(im1))
        else:
            im0 = to_tensor_(im0)
            im1 = to_tensor_(im1)
    else:
        if normalize:
            normalize_ = transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))
            im0 = normalize_(to_tensor_(im0))
            im1 = normalize_(to_tensor_(im1))
        else:
            im0 = to_tensor_(im0)
            im1 = to_tensor_(im1)

    return im0, im1