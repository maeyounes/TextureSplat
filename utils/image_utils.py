#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import numpy as np
import cv2


def compute_normal_mae_masked(pred_normal, gt_normal, alpha_mask):
    """Computes Mean Angular Error for normals using an alpha mask."""
    # NOTE: need to be modified later
    # Ensure inputs are on the same device and float type
    pred_normal = pred_normal.to(gt_normal.device).float()
    gt_normal = gt_normal.float()
    alpha_mask = alpha_mask.to(gt_normal.device)

    # Create a boolean mask (assuming alpha > 0.5 means foreground)
    # Ensure mask has the same spatial dimensions H, W
    if alpha_mask.dim() == 3: # C, H, W
        hard_mask = alpha_mask[0] > 0.5
    elif alpha_mask.dim() == 2: # H, W
        hard_mask = alpha_mask > 0.5
    else:
        raise ValueError("alpha_mask must have 2 or 3 dimensions (H, W or C, H, W)")
    

    # Check if there are any valid pixels in the mask
    if not torch.any(hard_mask):
        raise ValueError("No valid pixels in the mask")

    # Normalize vectors
    pred_normal_norm = F.normalize(pred_normal, p=2, dim=0)
    gt_normal_norm = F.normalize(gt_normal, p=2, dim=0)

    # Calculate dot product
    # Output shape: (H, W)
    dot_product = torch.sum(pred_normal_norm * gt_normal_norm, dim=0)
    # Clamp dot product to avoid numerical issues with acos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate angular error in degrees
    # Output shape: (H, W)
    angular_error = torch.acos(dot_product) * (180.0 / torch.pi)

    # Apply mask and calculate Mean Angular Error
    result_mae = torch.mean(angular_error * hard_mask.float())
    
            
    return result_mae.item()


def psnr(img1, img2):
    # mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    mse = (((img1 - img2) ** 2).reshape(img1.shape[0], -1)).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image

def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    colormap = matplotlib.colormaps['turbo']
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)


def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out