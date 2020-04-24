from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import matplotlib.cm     as cm
from torchvision import transforms as transforms
from torchvision import utils as vutils
from PIL import Image
from models import networks
from util import flow_utils
from skimage.transform import resize


# Get the input arguments here
parser = argparse.ArgumentParser(description='predict')
# parser.add_argument('--dataname', 
#                     default='Oxford',help='name of the dataset?')
parser.add_argument('--imglist',
                    default='./datafiles/sample_nighttime_data1_01.txt',
                    help='list containing the test images?')
parser.add_argument('--resultpath',
                    default='./results/nighttime_data1/',
                    help='where to save the results?')
parser.add_argument('--ckptpath',
                    default='./pretrained_ckpts/',
                    help='path to pretrained checkpoints?')
parser.add_argument('--epochno',
                    type=int, default=15,
                    help='which epoch checkpoints to load?')
args = parser.parse_args()


# Setup torch seed
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

image_paths = []
with open(args.imglist) as f:
    paths = f.readlines()
test_imgs0 = []
test_imgs1 =[]
for path in paths:
    path = path.strip().split(' ')
    test_imgs0.append(path[0])
    test_imgs1.append(path[1])
resultpath   = args.resultpath + '/'

# Check if the result directory exists, else make it
if not os.path.exists(resultpath):
    os.makedirs(resultpath)


# Setup the Generator(s) model(s) here
in_channels  = 3
out_channels = 3
is_grayscale = False
model_G_tst  = networks.define_G(input_nc=in_channels, 
                                 output_nc=out_channels, 
                                 ngf=32, 
                                 n_blocks=9, 
                                 n_blocks_shared=0, \
                                 n_domains=2, 
                                 norm='instance', 
                                 use_dropout=False, 
                                 gpu_ids=[0])
for i, net in enumerate(model_G_tst.networks):
    filename = args.ckptpath +'/' + str(args.epochno) + '_net_G'+ ('%d.pth' % i)
    net.load_state_dict(torch.load(filename))
model_G_tst.net_in_trainmode(0, False)
model_G_tst.net_in_trainmode(1, False)



# Setup the flowNet(s) model(s) here
from models.PWCNet import pwc_dc_net as PWCNet
model_flownet_ref = PWCNet(path=None)
model_flownet_tst = PWCNet(path=None)

model_flownet_ref = nn.DataParallel(model_flownet_ref, 
                                    device_ids=[0])
model_flownet_tst = nn.DataParallel(model_flownet_tst, 
                                    device_ids=[0])
model_flownet_ref.cuda()
model_flownet_tst.cuda()
# Ref flow network here
loadmodel_flownet_ref = args.ckptpath + '/' + str(args.epochno) + '_net_FlowNet1.pth'
# loadmodel_flownet_ref = './pre_flownets/pwc_net_chairs.pth.tar'
# Test flow network here
loadmodel_flownet_tst = args.ckptpath + '/' + str(args.epochno) + '_net_FlowNet0.pth'
if loadmodel_flownet_ref is not None:
    state_dict = torch.load(loadmodel_flownet_ref)
    model_flownet_ref.load_state_dict(state_dict['state_dict'])
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    # 	name = 'module.'+k # add `module.`
    # 	new_state_dict[name] = v
    # model_flownet_ref.load_state_dict(new_state_dict)
if loadmodel_flownet_tst is not None:
    state_dict = torch.load(loadmodel_flownet_tst)
    model_flownet_tst.load_state_dict(state_dict['state_dict'])
model_flownet_ref.eval()
model_flownet_tst.eval()


# Some functions here
def normalize():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])
def inv_normalize():
    return transforms.Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], 
                                std=[1/0.5, 1/0.5, 1/0.5])
def normalize_grayscale():
    return transforms.Normalize([0.5], [0.5])
def inv_normalize_grayscale():
    return transforms.Normalize([-0.5/0.5], [1/0.5])


# Test function 
def test(img0, img1):

    img0 = torch.FloatTensor(img0).cuda()
    img1 = torch.FloatTensor(img1).cuda()
    img0, img1 = Variable(img0), Variable(img1)

    # Prediction for reference flowarity network (netflowB)
    with torch.no_grad():
        output_ref, _, _, _, _ = model_flownet_ref(img0, img1)
    output_ref     = torch.squeeze(output_ref)
    pred_flow_ref  = output_ref.data.cpu().numpy().transpose(1, 2, 0)

    # Prediction for test flow network (netflowA)
    with torch.no_grad():
        encoded_img0    = model_G_tst.encode(img0, 0)
        encoded_img1    = model_G_tst.encode(img1, 0)
        rend_img0       = model_G_tst.decode(encoded_img0, 1)
        rend_img1       = model_G_tst.decode(encoded_img1, 1)
        rec_encoded_img0= model_G_tst.encode(rend_img0, 1)
        rec_encoded_img1= model_G_tst.encode(rend_img1, 1)
        rec_img0        = model_G_tst.decode(rec_encoded_img0, 0)
        rec_img1        = model_G_tst.decode(rec_encoded_img1, 0)
        output_tst, _, _, _, _ = model_flownet_tst(rec_img0, rec_img1)
    pred_flow_tst  = torch.squeeze(output_tst).cpu().numpy().transpose(1, 2, 0)

    return pred_flow_ref, pred_flow_tst


# Main function
def main():

    # Setup functions
    norm_  = normalize()
    resize_= transforms.Resize((256, 512), Image.BICUBIC)
    totens_= transforms.ToTensor()

    # Load, resize and normalize images
    assert(len(test_imgs0) == len(test_imgs1))
    numimgs = len(test_imgs0)
    for numimg in range(numimgs):
        test_img0 = test_imgs0[numimg]
        test_img1 = test_imgs1[numimg]
        print('Processing %s'% test_img0)
        if not is_grayscale:
            img0_o = Image.open(test_img0).convert('RGB')
            img1_o = Image.open(test_img1).convert('RGB')
        else:
            img0_o = Image.open(test_img0).convert('L')
            img1_o = Image.open(test_img1).convert('L')
        img0_  = resize_(img0_o)
        img1_  = resize_(img1_o)
        img0   = norm_(totens_(resize_(img0_o))).numpy()
        img1   = norm_(totens_(resize_(img1_o))).numpy()
        if not is_grayscale:
            img0   = np.reshape(img0, [1, 3, img0.shape[1], img0.shape[2]])
            img1   = np.reshape(img1, [1, 3, img1.shape[1], img1.shape[2]])
        else:
            img0   = np.reshape(img0, [1, 1, img0.shape[1], img0.shape[2]])
            img1   = np.reshape(img1, [1, 1, img1.shape[1], img1.shape[2]])

        # Generate results
        start_time = time.time()
        pred_flow_ref, pred_flow_tst  = test(img0, img1)

        pred_flow_ref_img = resize(flow_utils.flowToImg(pred_flow_ref).astype(np.float32) / 255.0,
                            [256, 512], mode='reflect').astype(np.float32)
        pred_flow_tst_img = resize(flow_utils.flowToImg(pred_flow_tst).astype(np.float32) / 255.0,
                            [256, 512], mode='reflect').astype(np.float32)

        print('Time taken = %.2f' % (time.time() - start_time))

        # Save the results
        saveresult = np.concatenate((totens_(img0_).numpy().transpose(1, 2, 0),
                                     pred_flow_ref_img,
                                     pred_flow_tst_img), axis=1)
        skimage.io.imsave(resultpath + 'result_' + str(numimg+1) + '.jpg',
                          (saveresult*255).astype('uint8'))
        flow_utils.writeFlow(resultpath + '/flow_ref_' + str(numimg+1) + '.flo', pred_flow_ref)
        flow_utils.writeFlow(resultpath + '/flow_tst_' + str(numimg+1) + '.flo', pred_flow_tst)


if __name__ == '__main__':
    main()