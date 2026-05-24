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
import torch.nn.functional as F
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.refl_utils import  get_specular_color_surfel, get_full_color_volume, get_full_color_volume_indirect, get_full_color_surfel
from utils.graphics_utils import linear_to_srgb, srgb_to_linear, rotation_between_z, init_predefined_omega
import numpy as np



def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe, rendered_features=None):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]

    # geometric normal from the splat orientation (transformed to world space)
    render_geom_normal = allmap[2:5]
    render_geom_normal = (render_geom_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    if rendered_features is None:
        # baseline: the rendered normal is the geometric (splat orientation) normal
        render_normal = render_geom_normal
    else:
        # textured stage: the rendered normal comes from the splatted per-primitive
        # texture normal map (already in world space; first 3 feature channels)
        render_normal = rendered_features[0:3]

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_geom_normal': render_geom_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal
    }



def render_initial(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_depth_median = regularizations['render_depth_median']
    render_depth_expected = regularizations['render_depth_expected']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb: 
        rendered_image = linear_to_srgb(rendered_image)
    final_image = rendered_image + bg_color[:, None, None] * (1 - render_alpha)

    rets =  {"render": final_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    }

    return rets



# with view dependent color for reflection setting
def render_surfel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None):

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    ## reflection strength 定义（即refl ratio）
    metallic = pc.get_metallic
    albedo = pc.get_albedo
    roughness = pc.get_rough

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # indirect light
    if pipe.use_asg:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        splat2world = pc.get_covariance(scaling_modifier)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        
        rotation_normal = rotation_between_z(normals).transpose(-1, -2)
        reflection_cartesian = (rotation_normal @ reflection[..., None])[..., 0]
        
        # import pdb;pdb.set_trace()
        omega, omega_la, omega_mu = pc.asg_param
        asg = pc.get_asg
        ep, la, mu = torch.split(asg, [3, 1, 1], dim=-1)
        
        Smooth = F.relu((reflection_cartesian[:, None] * omega[None]).sum(dim=-1, keepdim=True))

        ep = torch.exp(ep-3)
        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        exp_input = -la * (omega_la[None] * reflection_cartesian[:, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (omega_mu[None] * reflection_cartesian[:, None]).sum(dim=-1, keepdim=True).pow(2)
        indirect_asg = ep * Smooth * torch.exp(exp_input)
        indirect = indirect_asg.sum(dim=1).clamp_min(0.0)
    else:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        # import pdb;pdb.set_trace()
        shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        sh2indirect = eval_sh(3, shs_indirect, reflection)
        indirect = torch.clamp_min(sh2indirect, 0.0)
    

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        splat_features = torch.cat((metallic, roughness, albedo, indirect), dim=-1),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )


    base_color = rendered_image
    metallic = rendered_features[:1]
    roughness = rendered_features[1:2]
    albedo = rendered_features[2:5]
    indirect_light = rendered_features[5:8]


    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']


    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)
    
    if opt.indirect:
        specular, extra_dict = get_specular_color_surfel(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), metallic=metallic.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth, indirect_light=indirect_light.permute(1,2,0))
    else:
        specular, extra_dict = get_specular_color_surfel(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), metallic=metallic.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth)

    # Integrate the final image
    final_image = (1-metallic) * base_color + specular 
    
    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb: 
        final_image = linear_to_srgb(final_image)
        albedo = linear_to_srgb(albedo)
        specular = linear_to_srgb(specular)


    final_image = final_image + bg_color[:, None, None] * (1 - render_alpha)
    if opt.indirect:
        indirect_color = (1-metallic) * base_color + extra_dict['indirect_color']
        indirect_color = indirect_color + bg_color[:, None, None] * (1 - render_alpha)
        extra_dict['indirect_color'] = indirect_color

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "metallic_map": metallic,
            "diffuse_map": (1-metallic) * base_color,
            "specular_map": specular,
            "base_color_map": albedo,
            "roughness_map": roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal
    }
    
    if opt.indirect:
        results.update(extra_dict)



    return results




def render_volume(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    metallic = pc.get_metallic
    albedo = pc.get_albedo
    roughness = pc.get_rough

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)

    # indirect light
    if pipe.use_asg:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        splat2world = pc.get_covariance(scaling_modifier)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        
        rotation_normal = rotation_between_z(normals).transpose(-1, -2)
        reflection_cartesian = (rotation_normal @ reflection[..., None])[..., 0]
        
        # import pdb;pdb.set_trace()
        omega, omega_la, omega_mu = pc.asg_param
        asg = pc.get_asg
        ep, la, mu = torch.split(asg, [3, 1, 1], dim=-1)
        Smooth = F.relu((reflection_cartesian[:, None] * omega[None]).sum(dim=-1, keepdim=True))

        ep = torch.exp(ep-3)
        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        exp_input = -la * (omega_la[None] * reflection_cartesian[:, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (omega_mu[None] * reflection_cartesian[:, None]).sum(dim=-1, keepdim=True).pow(2)
        indirect_asg = ep * Smooth * torch.exp(exp_input)
        indirect = indirect_asg.sum(dim=1).clamp_min(0.0)

    else:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        # import pdb;pdb.set_trace()
        shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        sh2indirect = eval_sh(3, shs_indirect, reflection)
        indirect = torch.clamp_min(sh2indirect, 0.0)


    if opt.indirect:
        diffuse, specular, extra = get_full_color_volume_indirect(pc.get_envmap_2, means3D, albedo, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normals.contiguous(), opacity, metallic=metallic, roughness=roughness, pc=pc, indirect_light=indirect)
        visibility = extra['visibility']
        direct_light = extra["direct_light"]
    else: 
        diffuse, specular = get_full_color_volume(pc.get_envmap_2, means3D, albedo, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normals.contiguous(), opacity, metallic=metallic, roughness=roughness)
    colors_precomp = specular + diffuse




    if opt.indirect:
        features = torch.cat((roughness, metallic, diffuse, specular, albedo, visibility, indirect, direct_light), dim=-1)
    else:
        features = torch.cat((roughness, metallic, diffuse, specular, albedo), dim=-1)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        splat_features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )


    # get rendered diffuse color and other paras
    full_color = rendered_image     # (3,H,W)
    render_roughness = rendered_features[:1]   # (1,H,W)
    render_metallic = rendered_features[1:2]   # (1,H,W)
    render_diffuse_color = rendered_features[2:5]
    render_specular_color = rendered_features[5:8]
    render_albedo = rendered_features[8:11]  #
    if opt.indirect:
        render_visibility = rendered_features[11:12]
        render_indirect = rendered_features[12:15] 
        render_direct = rendered_features[15:18]     





    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']



    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb: 
        render_diffuse_color = linear_to_srgb(render_diffuse_color)
        render_specular_color = linear_to_srgb(render_specular_color)
        full_color = linear_to_srgb(full_color)

    final_image = full_color + bg_color[:, None, None] * (1 - render_alpha)
    



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "metallic_map": render_metallic,
            "diffuse_map": render_diffuse_color,
            "specular_map": render_specular_color,
            "base_color_map": render_albedo,
            "roughness_map": render_roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    }

    if opt.indirect:
        results.update(
            {
                "visibility": render_visibility,
                "indirect_light": render_indirect,
                "direct_light": render_direct
            }
        )

    return results




# Deferred shading with per-primitive texture maps (4th training stage).
# Material attributes and the tangent-space normal map are bilinearly sampled
# from each primitive's texture at the ray-splat intersection, splatted to
# screen-space buffers, then physically-based shaded.
def render_surfel_textured(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    # Hardware bilinear sampling is an eval-time opt-in driven by pipe.use_hw_sampling.
    # When True the rasterizer reads textures from a cached cudaTextureObject atlas
    # on the model via tex2D, instead of the software bilinear sampler.
    use_hw_sampling = bool(getattr(pipe, "use_hw_sampling", False))

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        use_hw_sampling=use_hw_sampling,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Per-primitive material/normal texture maps  -> (N, C, size, size)
    metallic = pc.get_metallic
    albedo = pc.get_albedo
    roughness = pc.get_rough
    texture_normal = pc.get_texture_normal

    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    # indirect light (per-primitive SH, evaluated in the reflection direction)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
    w_o = -dir_pp_normalized
    reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
    shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    sh2indirect = eval_sh(3, shs_indirect, reflection)
    indirect = torch.clamp_min(sh2indirect, 0.0)

    # SW path: feed the live [P, 8, T, T] tensor so autograd flows back to the
    # per-primitive texture parameters. HW path: skip the tensor (kernel reads
    # the cached atlas) and the atlas is sourced once from the model.
    if use_hw_sampling:
        texture_features_arg = torch.empty((means3D.shape[0], 0, 0, 0), device=means3D.device, dtype=torch.float32)
        atlas = pc.atlas
    else:
        texture_features_arg = torch.cat((texture_normal, metallic, roughness, albedo), dim=1)
        atlas = None

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        texture_features = texture_features_arg,
        splat_features = indirect,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        atlas = atlas,
    )

    base_color = rendered_image
    metallic = rendered_features[3:4]
    roughness = rendered_features[4:5]
    albedo = rendered_features[5:8]
    indirect_light = rendered_features[8:11]

    # 2DGS normal and regularizations (render_normal = splatted texture normal)
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe, rendered_features=rendered_features)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_geom_normal = regularizations['render_geom_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # Use the splatted texture normal map to perform the reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)

    if opt.indirect:
        specular, extra_dict = get_specular_color_surfel(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), metallic=metallic.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth, indirect_light=indirect_light.permute(1,2,0))
    else:
        specular, extra_dict = get_specular_color_surfel(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), metallic=metallic.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth)

    # Integrate the final image
    final_image = (1 - metallic) * base_color + specular

    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb:
        final_image = linear_to_srgb(final_image)
        albedo = linear_to_srgb(albedo)
        specular = linear_to_srgb(specular)

    final_image = final_image + bg_color[:, None, None] * (1 - render_alpha)
    if opt.indirect:
        indirect_color = (1 - metallic) * base_color + extra_dict['indirect_color']
        indirect_color = indirect_color + bg_color[:, None, None] * (1 - render_alpha)
        extra_dict['indirect_color'] = indirect_color

    results = {"render": final_image,
            "metallic_map": metallic,
            "diffuse_map": (1 - metallic) * base_color,
            "specular_map": specular,
            "base_color_map": albedo,
            "roughness_map": roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_geom_normal': render_geom_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal
    }

    if opt.indirect:
        results.update(extra_dict)

    return results
