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


# Get the input arguments here
parser = argparse.ArgumentParser(description='predict')
parser.add_argument('--dataname', 
                    default='Oxford',help='name of the dataset?')
parser.add_argument('--imgname',
                    default=None,help='name of the image?')
parser.add_argument('--datapath',
                    default=None,help='path to data?')
parser.add_argument('--resultpath',
                    default='./results/',help='where to save the results?')
parser.add_argument('--ckptpath',
                    default=None,help='path to pretrained checkpoints?')
args = parser.parse_args()


# Setup torch seed
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)


# Setup the inputs here
if args.dataname == 'Oxford':
    maxdisp = 48
else:
    raiseException('Provide maxdisp for your dataset and remove this line!')
imgname        = args.imgname
model          = 'stackhourglass'
test_left_img  = args.datapath + '/left/' + imgname+'.png'
test_right_img = args.datapath + '/right/' + imgname+'.png'
resultpath     = args.resultpath + '/' + imgname + '/'

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
    filename = args.ckptpath +'netG'+ ('%d.pth' % i)
    net.load_state_dict(torch.load(filename))
model_G_tst.net_in_trainmode(0, False)
model_G_tst.net_in_trainmode(1, False)



# Setup the DispNet(s) model(s) here
# Ref disparity network here
loadmodel_dispnet_ref = args.ckptpath + '/netDisp1.tar'
# Test disparity network here
loadmodel_dispnet_tst = args.ckptpath + '/netDisp0.pth'
if model == 'stackhourglass':
    from models.PSMNet_stackhourglass import PSMNet as dispnet
    model_dispnet_ref = dispnet(maxdisp, is_grayscale)
    model_dispnet_tst = dispnet(maxdisp, is_grayscale)
elif model == 'basic':
    from models.PSMNet_basic import PSMNet as dispnet
    model_dispnet_ref = dispnet(maxdisp, is_grayscale)
    model_dispnet_tst = dispnet(maxdisp, is_grayscale)
model_dispnet_ref = nn.DataParallel(model_dispnet_ref, 
                                    device_ids=[0])
model_dispnet_tst = nn.DataParallel(model_dispnet_tst, 
                                    device_ids=[0])
model_dispnet_ref.cuda()
model_dispnet_tst.cuda()
if loadmodel_dispnet_ref is not None:
    state_dict = torch.load(loadmodel_dispnet_ref)
    model_dispnet_ref.load_state_dict(state_dict['state_dict'])
if loadmodel_dispnet_tst is not None:
    state_dict = torch.load(loadmodel_dispnet_tst)
    model_dispnet_tst.load_state_dict(state_dict['state_dict'])
model_dispnet_ref.eval()
model_dispnet_tst.eval()


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
def test(imgL, imgR):

    imgL = torch.FloatTensor(imgL).cuda()
    imgR = torch.FloatTensor(imgR).cuda()
    imgL, imgR = Variable(imgL), Variable(imgR)

    # Prediction for reference disparity network (netDispB)
    with torch.no_grad():
        _, _, output_ref = model_dispnet_ref(imgL, imgR)
    output_ref     = torch.squeeze(output_ref)
    pred_disp_ref  = output_ref.data.cpu().numpy()

    # Prediction for test disparity network (netDispA)
    with torch.no_grad():
        encoded_imgL    = model_G_tst.encode(imgL, 0)
        encoded_imgR    = model_G_tst.encode(imgR, 0)
        rend_imgL       = model_G_tst.decode(encoded_imgL, 1)
        rend_imgR       = model_G_tst.decode(encoded_imgR, 1)
        rec_encoded_imgL= model_G_tst.encode(rend_imgL, 1)
        rec_encoded_imgR= model_G_tst.encode(rend_imgR, 1)
        rec_imgL        = model_G_tst.decode(rec_encoded_imgL, 0)
        rec_imgR        = model_G_tst.decode(rec_encoded_imgR, 0)
        _, _, output_tst= model_dispnet_tst(rec_imgL, rec_imgR)
    pred_disp_tst  = torch.squeeze(output_tst).data.cpu().numpy()

    return pred_disp_ref, pred_disp_tst


# Main function
def main():

    # Setup functions
    norm_  = normalize()
    resize_= transforms.Resize((256, 512), Image.BICUBIC)
    totens_= transforms.ToTensor()
    topil_ = transforms.ToPILImage()
    cm_jet_= cm.get_cmap('jet')

    # Load, resize and normalize images
    if not is_grayscale:
        imgL_o = Image.open(test_left_img).convert('RGB')
        imgR_o = Image.open(test_right_img).convert('RGB')
    else:
        imgL_o = Image.open(test_left_img).convert('L')
        imgR_o = Image.open(test_right_img).convert('L')
    imgL_  = resize_(imgL_o)
    imgR_  = resize_(imgR_o)
    imgL   = norm_(totens_(resize_(imgL_o))).numpy()
    imgR   = norm_(totens_(resize_(imgR_o))).numpy()
    if not is_grayscale:
        imgL   = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR   = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])
    else:
        imgL   = np.reshape(imgL, [1, 1, imgL.shape[1], imgL.shape[2]])
        imgR   = np.reshape(imgR, [1, 1, imgR.shape[1], imgR.shape[2]])

    # Generate results
    start_time = time.time()
    pred_disp_ref, pred_disp_tst  = test(imgL, imgR)
    print('time = %.2f' % (time.time() - start_time))

    # Save the results
    imgL_.save(resultpath + '/img_left.jpg')
    imgR_.save(resultpath + '/img_right.jpg')
    skimage.io.imsave(resultpath + '/disp_ref.png', (pred_disp_ref * 256).astype('uint16'))
    skimage.io.imsave(resultpath + '/graydisp_ref.png', ((pred_disp_ref/maxdisp)*255.0).astype('uint16'))
    skimage.io.imsave(resultpath + '/disp_tst.png', (pred_disp_tst * 256).astype('uint16'))
    skimage.io.imsave(resultpath + '/graydisp_tst.png', ((pred_disp_tst/maxdisp)*255.0).astype('uint16'))


if __name__ == '__main__':
    main()






