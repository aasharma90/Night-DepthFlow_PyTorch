import torch
import torch.nn.functional as F
from torch import nn


def EPE(input_flow, target_flow, sparse=False, mean=True):
    # print('Input flow: ', torch.min(input_flow), torch.max(input_flow),
    #       torch.mean(input_flow))
    # print('Target flow: ', torch.min(target_flow), torch.max(target_flow),
    #       torch.mean(target_flow))
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def bad_pixel(input_flow, target_flow, sparse=False):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    target_mag = torch.norm(target_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] != 0) & (target_flow[:,1] != 0) & (EPE_map>3) & ((EPE_map/target_mag)>0.05)

    else:
        mask = (EPE_map>45) & ((EPE_map/target_mag)>0.09)
    return mask.sum()/batch_size

def bad_pixel_rate(input_flow, target_flow, sparse=False):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    target_mag = torch.norm(target_flow,2,1)
    batch_size = EPE_map.size(0)
        # invalid flow is defined with both flow coordinates to be exactly 0
    mask = (target_flow[:,0] != 0) & (target_flow[:,1] != 0) & (EPE_map>3) & ((EPE_map/target_mag)>0.05)
    total_mask = (target_flow[:,0] != 0) & (target_flow[:,1] != 0)

    bad_p = mask.sum()/batch_size   
    total_p = total_mask.sum()/batch_size 
    bad_p_r = bad_p.float()/total_p.float() 

    # print(bad_p)
    # print(total_p)
    # print(bad_p_r)
    # stop
    # print(total_p)
    if total_p == 0:
        bad_p_r = total_p

    return bad_p_r

def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.upsample(target, (h, w), mode='bilinear')
        return EPE(output, target_scaled, sparse, mean=False)
        #return nn.MSELoss()(output, target_scaled)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss

def multiscaleEPE_scaled(network_output, target_flow, weights=None, sparse=False, scaled=1):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.upsample(target, (h, w), mode='bilinear')
        return EPE(output, target_scaled, sparse, mean=False)
        #return nn.MSELoss()(output, target_scaled)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output*scaled, target_flow, sparse)
    return loss

def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.upsample(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)
def realBadPixel(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.upsample(output, (h,w), mode='bilinear', align_corners=False)
    return bad_pixel(upsampled_output, target, sparse)    
def realBadPixelRate(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.upsample(output, (h,w), mode='bilinear', align_corners=False)
    return bad_pixel_rate(upsampled_output, target, sparse)      
# def realEPE(output, target, sparse=False):
#     b, _, h, w = output.size()
#     target_scaled = F.interpolate(target, (h,w), mode='bilinear')
#     return EPE(output, target_scaled, sparse, mean=True)

def multiscaleEPE_uncertainty(network_output, target_flow, uncertainty_maps, weights=None, sparse=False, scaled=20):
    def one_scale(output, target, uncertainty, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
            uncertainty_scaled = F.interpolate(uncertainty, (h, w), mode='bilinear')
        else:
            target_scaled = F.interpolate(target, (h, w), mode='bilinear')
            uncertainty_scaled = F.interpolate(uncertainty, (h, w), mode='bilinear')
        return EPE_uncertainty(output, target_scaled, uncertainty_scaled, sparse, mean=False)
        #return nn.MSELoss()(output, target_scaled)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output*scaled, target_flow, uncertainty_maps, sparse)
    return loss    

def EPE_uncertainty(input_flow, target_flow, uncertainty_maps, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    certainty_maps = torch.mean(uncertainty_maps,dim=1)
    # certainty_maps = certainty_maps - certainty_maps.min()
    # certainty_maps = certainty_maps/certainty_maps.max()
    EPE_map = EPE_map * certainty_maps
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size    