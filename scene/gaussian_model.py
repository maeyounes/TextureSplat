import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from cubemapencoder import CubemapEncoder
from scene.light import EnvLight
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, init_predefined_omega
from utils.general_utils import strip_symmetric, build_scaling_rotation, safe_normalize, flip_align_view
from utils.refl_utils import sample_camera_rays, get_env_rayd1, get_env_rayd2
import raytracing


def get_env_direction1(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'),
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi,
        costheta,
        -sintheta*cosphi
        ), dim=-1)
    return env_directions


def get_env_direction2(H, W):
    gx, gy = torch.meshgrid(
        torch.linspace(-torch.pi, torch.pi, W, device='cuda'),
        torch.linspace(0, torch.pi, H, device='cuda'),
        indexing='xy'
    )
    env_directions = torch.stack((
        torch.sin(gy)*torch.cos(gx),
        torch.sin(gy)*torch.sin(gx),
        torch.cos(gy)
    ), dim=-1)
    return env_directions


class GaussianModel:
    """Baseline reflective 2DGS model: one PBR attribute per primitive.

    Per-primitive PBR attributes follow a metallic workflow:
      - ``_albedo``   : base colour / albedo (specular F0 tint).
      - ``_metallic`` : metalness in [0, 1] (0 = dielectric, 1 = metal).
      - ``_roughness``: surface roughness in [0, 1].
    The diffuse term is provided by the view-dependent SH colour
    (``_features_dc`` / ``_features_rest``).
    """

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.metallic_activation = torch.sigmoid
        self.inverse_metallic_activation = inverse_sigmoid

        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.asg_param = init_predefined_omega(4, 8)


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._metallic = torch.empty(0)
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._indirect_dc = torch.empty(0)
        self._indirect_rest = torch.empty(0)
        self._indirect_asg = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.free_radius = 0
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.init_metallic_value = 0.01
        self.init_roughness_value = 0.1 #[0,1]
        self.enlarge_scale = 1.5
        self.metallic_msk_thr = 0.02
        self.rough_msk_thr = 0.1

        self.env_map = None
        self.env_map_2 = None
        self.env_H, self.env_W = 256, 512
        self.env_directions1 = get_env_direction1(self.env_H, self.env_W)
        self.env_directions2 = get_env_direction2(self.env_H, self.env_W)
        self.ray_tracer = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._metallic,
            self._roughness,
            self._albedo,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._indirect_asg,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._metallic,
        self._roughness,
        self._albedo,
        self._features_dc,
        self._features_rest,
        self._indirect_dc,
        self._indirect_rest,
        self._indirect_asg,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self._indirect_asg = nn.Parameter(torch.zeros(self._rotation.shape[0], 32, 5, device='cuda').requires_grad_(True))
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)

    def set_opacity_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group['lr'] = lr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)

    @property
    def get_rough(self):
        return self.roughness_activation(self._roughness)

    @property
    def get_albedo(self):
        return self.color_activation(self._albedo)

    def get_normal(self, scaling_modifier, dir_pp_normalized):
        splat2world = self.get_covariance(scaling_modifier)
        normals_raw = splat2world[:, 2, :3]
        normals_raw, _ = flip_align_view(normals_raw, dir_pp_normalized)
        return safe_normalize(normals_raw)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_indirect(self):
        indirect_dc = self._indirect_dc
        indirect_rest = self._indirect_rest
        return torch.cat((indirect_dc, indirect_rest), dim=1)

    @property
    def get_asg(self):
        return self._indirect_asg

    def render_env_map(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map(directions1, mode="pure_env"), 'env2': self.env_map(directions2, mode="pure_env")}

    def render_env_map_2(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map_2(directions1, mode="pure_env"), 'env2': self.env_map_2(directions2, mode="pure_env")}

    @property
    def get_envmap(self):
        return self.env_map

    @property
    def get_envmap_2(self):
        return self.env_map_2

    @property
    def get_metallic_ratio(self):
        metallic = self.get_metallic
        return (metallic > 0.1).sum() / metallic.shape[0]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        sh_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        sh_features[:, :3, 0 ] = fused_color
        sh_features[:, 3:, 1:] = 0.0
        sh_indirect = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        asg_indirect = torch.zeros((fused_color.shape[0], 5, 32)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        metallic = self.inverse_metallic_activation(torch.ones_like(opacities).cuda() * self.init_metallic_value)
        roughness = self.inverse_roughness_activation(torch.ones_like(opacities).cuda() * self.init_roughness_value)

        def initialize_albedo(point_cloud, init_color=0.5, noise_level=0.05):
            base_color = torch.full((point_cloud.shape[0], 3), init_color, dtype=torch.float, device="cuda")
            noise = (torch.rand(point_cloud.shape[0], 3, dtype=torch.float, device="cuda") - 0.5) * noise_level
            return torch.clamp(base_color + noise, 0.0, 1.0)

        albedo = self.inverse_color_activation(initialize_albedo(fused_point_cloud))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(sh_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(sh_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_dc = nn.Parameter(sh_indirect[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(sh_indirect[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_asg = nn.Parameter(asg_indirect.transpose(1, 2).contiguous().requires_grad_(True))

        self.env_map = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
        self.env_map_2 = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},
            {'params': self.env_map_2.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env2"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._indirect_dc], 'lr': training_args.indirect_lr, "name": "ind_dc"},
            {'params': [self._indirect_rest], 'lr': training_args.indirect_lr / 20.0, "name": "ind_rest"},
            {'params': [self._indirect_asg], 'lr': training_args.asg_lr, "name": "ind_asg"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._indirect_dc.shape[1]*self._indirect_dc.shape[2]):
            l.append('ind_dc_{}'.format(i))
        for i in range(self._indirect_rest.shape[1]*self._indirect_rest.shape[2]):
            l.append('ind_rest_{}'.format(i))
        for i in range(self._indirect_asg.shape[1]*self._indirect_asg.shape[2]):
            l.append('ind_asg_{}'.format(i))
        l.append('opacity')
        l.append('metallic')
        l.append('roughness')
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_dc = self._indirect_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_rest = self._indirect_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_asg = self._indirect_asg.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        metallic = self._metallic.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, f_dc, f_rest, ind_dc, ind_rest, ind_asg, opacities, metallic, roughness, albedo, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.env_map is not None:
            torch.save(self.env_map.state_dict(), path.replace('.ply', '1.map'))

        if self.env_map_2 is not None:
            torch.save(self.env_map_2.state_dict(), path.replace('.ply', '2.map'))


    def reset_opacity0(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1(self, exclusive_msk = None):
        RESET_V = 0.9
        opacity_old = self.get_opacity
        o_msk = (opacity_old > RESET_V).flatten()
        if exclusive_msk is not None:
            o_msk = torch.logical_or(o_msk, exclusive_msk)
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([RESET_V]).cuda())
        opacities_new[o_msk] = self._opacity[o_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity1_strategy2(self):
        RESET_B = 1.5
        opacity_old = self.get_opacity
        opacities_new = inverse_sigmoid((opacity_old*RESET_B).clamp(0,0.99))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]


    def reset_metallic(self, exclusive_msk = None):
        metallic_new = inverse_sigmoid(torch.max(self.get_metallic, torch.ones_like(self.get_metallic)*self.init_metallic_value))
        if exclusive_msk is not None:
            metallic_new[exclusive_msk] = self._metallic[exclusive_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(metallic_new, "metallic")
        if "metallic" not in optimizable_tensors: return
        self._metallic = optimizable_tensors["metallic"]


    def dist_rot(self):
        metallic_msk = self.get_metallic.flatten() > self.metallic_msk_thr
        rot = self.get_rotation.clone()
        dist_rot = self.rotation_activation(rot + torch.randn_like(rot)*0.08)
        dist_rot[metallic_msk] = rot[metallic_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_rot, "rotation")
        if "rotation" not in optimizable_tensors: return
        self._rotation = optimizable_tensors["rotation"]

    def dist_albedo(self, exclusive_msk = None):
        DIST_RANGE = 0.4
        metallic_msk = self.get_metallic.flatten() > self.metallic_msk_thr
        if exclusive_msk is not None:
            metallic_msk = torch.logical_or(metallic_msk, exclusive_msk)
        dcc = self._albedo.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE)
        dist_dcc[metallic_msk] = dcc[metallic_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "albedo")
        if "albedo" not in optimizable_tensors: return
        self._albedo = optimizable_tensors["albedo"]

    def dist_color(self, exclusive_msk = None):
        DIST_RANGE = 0.4
        metallic_msk = self.get_metallic.flatten() > self.metallic_msk_thr
        if exclusive_msk is not None:
            metallic_msk = torch.logical_or(metallic_msk, exclusive_msk)
        dcc = self._features_dc.clone()
        dist_dcc = dcc + (torch.rand_like(dcc)*DIST_RANGE*2-DIST_RANGE)
        dist_dcc[metallic_msk] = dcc[metallic_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(dist_dcc, "f_dc")
        if "f_dc" not in optimizable_tensors: return
        self._features_dc = optimizable_tensors["f_dc"]

    def enlarge_metallic_scales(self, ret_raw=True, exclusive_msk=None):
        ENLARGE_SCALE = self.enlarge_scale
        METALLIC_MSK_THR = self.metallic_msk_thr
        ROUGH_MSK_THR = self.rough_msk_thr

        metallic_msk = self.get_metallic.flatten() < METALLIC_MSK_THR
        rough_msk = self.get_rough.flatten() > ROUGH_MSK_THR
        combined_msk = torch.logical_or(metallic_msk, rough_msk)
        if exclusive_msk is not None:
            combined_msk = torch.logical_or(combined_msk, exclusive_msk)
        scales = self.get_scaling
        rmin_axis = (torch.ones_like(scales) * ENLARGE_SCALE)
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scales * rmin_axis)
            scale_new[combined_msk] = self._scaling[combined_msk]
        else:
            scale_new = scales * rmin_axis
            scale_new[combined_msk] = scales[combined_msk]
        return scale_new

    def reset_scale(self, exclusive_msk = None):
        scale_new = self.enlarge_metallic_scales(ret_raw=True, exclusive_msk=exclusive_msk)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        if "scaling" not in optimizable_tensors: return
        self._scaling = optimizable_tensors["scaling"]


    def reset_features(self, reset_value_dc=0.0, reset_value_rest=0.0):
        features_dc_new = torch.full_like(self._features_dc, reset_value_dc, dtype=torch.float, device="cuda")
        features_rest_new = torch.full_like(self._features_rest, reset_value_rest, dtype=torch.float, device="cuda")

        optimizable_tensors = self.replace_tensor_to_optimizer(features_dc_new, "f_dc")
        optimizable_tensors.update(self.replace_tensor_to_optimizer(features_rest_new, "f_rest"))
        self.active_sh_degree = 0

        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]


    def reset_albedo(self, reset_value=0.5, noise_level=0.05):
        base_color = torch.full_like(self._albedo, reset_value, dtype=torch.float, device="cuda")
        noise = (torch.rand_like(base_color, dtype=torch.float, device="cuda") - 0.5) * noise_level
        albedo_new = torch.clamp(base_color + noise, 0.0, 1.0)
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_color_activation(albedo_new), "albedo")
        if "albedo" in optimizable_tensors:
            self._albedo = optimizable_tensors["albedo"]

    def reset_metallic_value(self, reset_value=0.01):
        metallic_new = torch.full_like(self._metallic, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_metallic_activation(metallic_new), "metallic")
        if "metallic" in optimizable_tensors:
            self._metallic = optimizable_tensors["metallic"]

    def reset_roughness(self, reset_value=0.1):
        roughness_new = torch.full_like(self._roughness, reset_value, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(self.inverse_roughness_activation(roughness_new), "roughness")
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]


    def load_ply(self, path, relight=False, args=None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

        albedo = np.stack((np.asarray(plydata.elements[0]['albedo_0']),
                           np.asarray(plydata.elements[0]['albedo_1']),
                           np.asarray(plydata.elements[0]['albedo_2'])),  axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        self.active_sh_degree = self.max_sh_degree

        indirect_dc = np.zeros((xyz.shape[0], 3, 1))
        indirect_dc[:, 0, 0] = np.asarray(plydata.elements[0]["ind_dc_0"])
        indirect_dc[:, 1, 0] = np.asarray(plydata.elements[0]["ind_dc_1"])
        indirect_dc[:, 2, 0] = np.asarray(plydata.elements[0]["ind_dc_2"])

        extra_ind_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_rest_")]
        extra_ind_names = sorted(extra_ind_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_ind_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        indirect_extra = np.zeros((xyz.shape[0], len(extra_ind_names)))
        for idx, attr_name in enumerate(extra_ind_names):
            indirect_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        indirect_extra = indirect_extra.reshape((indirect_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        extra_asg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_asg_")]
        extra_asg_names = sorted(extra_asg_names, key = lambda x: int(x.split('_')[-1]))
        indirect_asg = np.zeros((xyz.shape[0], len(extra_asg_names)))
        for idx, attr_name in enumerate(extra_asg_names):
            indirect_asg[:, idx] = np.asarray(plydata.elements[0][attr_name])
        indirect_asg = indirect_asg.reshape((indirect_asg.shape[0], 5, -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if not relight:
            map_path1 = path.replace('.ply', '1.map')
            map_path2 = path.replace('.ply', '2.map')
            if os.path.exists(map_path1)  and os.path.exists(map_path2):
                self.env_map = EnvLight(path=None, device='cuda',  max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map.load_state_dict(torch.load(map_path1))
                self.env_map.build_mips()
                self.env_map_2 = EnvLight(path=None, device='cuda',  max_res=args.envmap_max_res, min_roughness=args.envmap_min_roughness, max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map_2.load_state_dict(torch.load(map_path2))
                self.env_map_2.build_mips()
        else:
            map_path = path.replace('.ply', '.hdr')
            self.env_map = EnvLight(path=map_path, device='cuda', trainable=True).cuda()


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self._indirect_dc = nn.Parameter(torch.tensor(indirect_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(torch.tensor(indirect_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_asg = nn.Parameter(torch.tensor(indirect_asg, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env" or group["name"] == "env2": continue   # #
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._metallic = optimizable_tensors['metallic']
        self._albedo = optimizable_tensors['albedo']
        self._roughness = optimizable_tensors['roughness']
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._indirect_asg = optimizable_tensors["ind_asg"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "mlp" or group["name"] == "env" or group["name"] == "env2": continue   # #
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_metallic, new_roughness, new_albedo, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "metallic": new_metallic,
        "roughness": new_roughness,
        "albedo": new_albedo,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "ind_dc": new_indirect_dc,
        "ind_rest": new_indirect_rest,
        "ind_asg": new_indirect_asg,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._metallic = optimizable_tensors['metallic']
        self._roughness = optimizable_tensors['roughness']
        self._albedo = optimizable_tensors['albedo']
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._indirect_dc = optimizable_tensors["ind_dc"]
        self._indirect_rest = optimizable_tensors["ind_rest"]
        self._indirect_asg = optimizable_tensors["ind_asg"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)

        new_indirect_dc = self._indirect_dc[selected_pts_mask].repeat(N,1,1)
        new_indirect_rest = self._indirect_rest[selected_pts_mask].repeat(N,1,1)
        new_indirect_asg = self._indirect_asg[selected_pts_mask].repeat(N,1,1)

        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_metallic, new_roughness, new_albedo, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        new_indirect_dc = self._indirect_dc[selected_pts_mask]
        new_indirect_rest = self._indirect_rest[selected_pts_mask]
        new_indirect_asg = self._indirect_asg[selected_pts_mask]

        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_metallic, new_roughness, new_albedo, new_features_dc, new_features_rest, new_indirect_dc, new_indirect_asg, new_indirect_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)  # #
        self.denom[update_filter] += 1

    def set_requires_grad(self, attrib_name, state: bool):
        getattr(self, f"_{attrib_name}").requires_grad = state

    def update_mesh(self, mesh):
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        self.ray_tracer = raytracing.RayTracer(vertices, faces)

    def load_mesh_from_ply(self, model_path, iteration):
        import open3d as o3d
        import os

        ply_path = os.path.join(model_path, f'test_{iteration:06d}.ply')
        mesh = o3d.io.read_triangle_mesh(ply_path)
        self.update_mesh(mesh)


class TexturedGaussianModel(GaussianModel):
    """Gaussian model with per-primitive texture maps for the material attributes
    (albedo, metallic, roughness) and a tangent-space normal map.

    Used for the 4th, deferred-shading training stage. It is created from a
    trained baseline :class:`GaussianModel` via :meth:`from_baseline` or
    :meth:`restore_from_pretrained` (from a saved baseline checkpoint). The
    scalar material attributes of the baseline are broadcast to constant texture
    maps as the starting point for texture optimisation.

    Geometry / SH / indirect-SH / environment maps are inherited unchanged from
    :class:`GaussianModel`; only the texture-specific tensors and the methods
    that touch them are overridden here.
    """

    def __init__(self, sh_degree: int, texture_size: int = 2):
        self._texture_size = texture_size
        self._texture_normal = torch.empty(0)
        # Cached per-primitive texture atlas for hardware bilinear sampling.
        # Built lazily on first read of `self.atlas` (eval-time only); geometry
        # and texture parameter counts are frozen at this stage so a single
        # build is enough for the model's lifetime.
        self._atlas = None
        super().__init__(sh_degree)

    @property
    def atlas(self):
        """Cached per-primitive ``TextureAtlas`` for the HW bilinear path.

        Built on first access from the same channel-planar [P, C, T, T] tensor
        the software path samples (channel order:
        normal(3) | metallic(1) | roughness(1) | albedo(3)). Eval-time only —
        do not access during training (textures change every step).
        """
        if self._atlas is None:
            from diff_surfel_rasterization import TextureAtlas
            features = torch.cat(
                (self.get_texture_normal, self.get_metallic,
                 self.get_rough, self.get_albedo),
                dim=1,
            ).contiguous().to(torch.float32)
            self._atlas = TextureAtlas(features)
        return self._atlas

    def setup_functions(self):
        super().setup_functions()

        # Two-component tangent-normal parametrisation: store (x, y), derive a
        # positive z. Keeps the sampled normal close to the unit sphere.
        def normal_texture_activation(x):
            if x.dim() == 4:
                orig_shape = x.shape
                x = x.permute(0, 2, 3, 1).reshape(-1, 2)
                xy = torch.tanh(x)
                xy_squared_norm = torch.sum(xy ** 2, dim=1, keepdim=True).clamp(max=0.99)
                z = torch.sqrt(1 - xy_squared_norm)
                normals = torch.cat([xy, z], dim=1)
                return normals.reshape(orig_shape[0], orig_shape[2], orig_shape[3], 3).permute(0, 3, 1, 2)
            else:
                xy = torch.tanh(x)
                xy_squared_norm = torch.sum(xy ** 2, dim=1, keepdim=True).clamp(max=0.99)
                z = torch.sqrt(1 - xy_squared_norm)
                return torch.cat([xy, z], dim=1)

        self.normal_texture_activation = normal_texture_activation

    @property
    def get_texture_normal(self):
        return self.normal_texture_activation(self._texture_normal)

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._metallic,
            self._roughness,
            self._albedo,
            self._features_dc,
            self._features_rest,
            self._indirect_dc,
            self._indirect_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._texture_normal,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args, checkpoint_path=None, dataset=None):
        """Resume from a textured-model checkpoint (18-element capture tuple)."""
        (self.active_sh_degree,
         self._xyz,
         self._metallic,
         self._roughness,
         self._albedo,
         self._features_dc,
         self._features_rest,
         self._indirect_dc,
         self._indirect_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self._texture_normal,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self._texture_size = self._texture_normal.shape[-1]
        if checkpoint_path is not None:
            self._load_envmaps_from_checkpoint(checkpoint_path, dataset)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    # ------------------------------------------------------------------ #
    # Transfer from a baseline model
    # ------------------------------------------------------------------ #
    def _broadcast_scalars_to_textures(self, metallic_raw, rough_raw, albedo_raw):
        """Broadcast per-primitive scalar attributes (raw, pre-activation) to
        constant [N, C, size, size] texture maps."""
        s = self._texture_size
        def to_tex(t):
            return nn.Parameter(t.detach().unsqueeze(-1).unsqueeze(-1)
                                .repeat(1, 1, s, s).contiguous().requires_grad_(True))
        return to_tex(metallic_raw), to_tex(rough_raw), to_tex(albedo_raw)

    @classmethod
    def from_baseline(cls, base: GaussianModel, training_args, texture_size: int):
        """Build a TexturedGaussianModel from an in-memory trained baseline model.

        Geometry / SH / indirect SH / environment maps are shared by reference;
        the scalar material attributes are broadcast into texture maps and the
        tangent normal map is zero-initialised.
        """
        tex = cls(base.max_sh_degree, texture_size=texture_size)
        tex.active_sh_degree = base.active_sh_degree
        tex.spatial_lr_scale = base.spatial_lr_scale

        # Geometry, appearance and indirect lighting are carried over directly.
        tex._xyz = base._xyz
        tex._scaling = base._scaling
        tex._rotation = base._rotation
        tex._opacity = base._opacity
        tex._features_dc = base._features_dc
        tex._features_rest = base._features_rest
        tex._indirect_dc = base._indirect_dc
        tex._indirect_rest = base._indirect_rest
        tex.max_radii2D = base.max_radii2D

        # Environment maps are shared by reference (in-memory transfer).
        tex.env_map = base.env_map
        tex.env_map_2 = base.env_map_2

        # Runtime attributes / mesh ray tracer.
        tex.init_metallic_value = base.init_metallic_value
        tex.init_roughness_value = base.init_roughness_value
        tex.enlarge_scale = base.enlarge_scale
        tex.metallic_msk_thr = base.metallic_msk_thr
        tex.rough_msk_thr = base.rough_msk_thr
        tex.ray_tracer = base.ray_tracer

        # Texture maps: zero tangent normals, scalar materials broadcast.
        N = base._xyz.shape[0]
        tex._texture_normal = nn.Parameter(torch.zeros(
            (N, 2, texture_size, texture_size), dtype=torch.float, device="cuda").requires_grad_(True))
        tex._metallic, tex._roughness, tex._albedo = tex._broadcast_scalars_to_textures(
            base._metallic.data, base._roughness.data, base._albedo.data)

        tex.training_setup(training_args)
        return tex

    def _load_envmaps_from_checkpoint(self, checkpoint_path, dataset):
        """Load the two environment maps written next to a checkpoint by
        ``scene.save`` (``point_cloud/iteration_<it>/point_cloud{1,2}.map``).
        `dataset` is the ModelParams group, used for the EnvLight resolution."""
        try:
            checkpoint_iteration = int(os.path.basename(checkpoint_path).split('.')[0].split('chkpnt')[-1])
        except ValueError:
            raise ValueError(f"Cannot parse iteration from checkpoint name '{checkpoint_path}'")
        envmap_dir = os.path.join(os.path.dirname(checkpoint_path), 'point_cloud',
                                  f'iteration_{checkpoint_iteration}')
        map_path1 = os.path.join(envmap_dir, 'point_cloud1.map')
        map_path2 = os.path.join(envmap_dir, 'point_cloud2.map')
        if not (os.path.exists(map_path1) and os.path.exists(map_path2)):
            raise FileNotFoundError(
                f"Environment maps not found next to checkpoint: expected "
                f"{map_path1} and {map_path2}. Make sure the checkpoint iteration "
                f"matches a saved point cloud (scene.save).")
        self.env_map = EnvLight(path=None, device='cuda', max_res=dataset.envmap_max_res,
                                min_roughness=dataset.envmap_min_roughness,
                                max_roughness=dataset.envmap_max_roughness, trainable=True).cuda()
        self.env_map.load_state_dict(torch.load(map_path1))
        self.env_map.build_mips()
        self.env_map_2 = EnvLight(path=None, device='cuda', max_res=dataset.envmap_max_res,
                                  min_roughness=dataset.envmap_min_roughness,
                                  max_roughness=dataset.envmap_max_roughness, trainable=True).cuda()
        self.env_map_2.load_state_dict(torch.load(map_path2))
        self.env_map_2.build_mips()

    def restore_from_pretrained(self, model_args, training_args, checkpoint_path, dataset):
        """Bootstrap the textured model from a saved BASELINE checkpoint
        (18-element :meth:`GaussianModel.capture` tuple)."""
        (self.active_sh_degree,
         self._xyz,
         metallic,
         roughness,
         albedo,
         self._features_dc,
         self._features_rest,
         self._indirect_dc,
         self._indirect_rest,
         _indirect_asg,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         _opt_dict,
         self.spatial_lr_scale) = model_args

        N = self._xyz.shape[0]
        self._texture_normal = nn.Parameter(torch.zeros(
            (N, 2, self._texture_size, self._texture_size), dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic, self._roughness, self._albedo = self._broadcast_scalars_to_textures(
            metallic.data, roughness.data, albedo.data)

        self._load_envmaps_from_checkpoint(checkpoint_path, dataset)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},
            {'params': self.env_map_2.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env2"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._indirect_dc], 'lr': training_args.indirect_lr, "name": "ind_dc"},
            {'params': [self._indirect_rest], 'lr': training_args.indirect_lr / 20.0, "name": "ind_rest"},
            {'params': [self._texture_normal], 'lr': training_args.normal_lr, "name": "texture_normal"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    # ------------------------------------------------------------------ #
    # PLY serialisation
    # ------------------------------------------------------------------ #
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._indirect_dc.shape[1] * self._indirect_dc.shape[2]):
            l.append('ind_dc_{}'.format(i))
        for i in range(self._indirect_rest.shape[1] * self._indirect_rest.shape[2]):
            l.append('ind_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._texture_normal.shape[1] * self._texture_normal.shape[2] * self._texture_normal.shape[3]):
            l.append('tex_n_{}'.format(i))
        for i in range(self._metallic.shape[1] * self._metallic.shape[2] * self._metallic.shape[3]):
            l.append('metallic_{}'.format(i))
        for i in range(self._albedo.shape[1] * self._albedo.shape[2] * self._albedo.shape[3]):
            l.append('albedo_{}'.format(i))
        for i in range(self._roughness.shape[1] * self._roughness.shape[2] * self._roughness.shape[3]):
            l.append('roughness_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_dc = self._indirect_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ind_rest = self._indirect_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        texture_normal = self._texture_normal.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        roughness = self._roughness.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        albedo = self._albedo.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        metallic = self._metallic.detach().flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, ind_dc, ind_rest, opacities, scale, rotation,
                                     texture_normal, metallic, albedo, roughness), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.env_map is not None:
            torch.save(self.env_map.state_dict(), path.replace('.ply', '1.map'))
        if self.env_map_2 is not None:
            torch.save(self.env_map_2.state_dict(), path.replace('.ply', '2.map'))

    def load_ply(self, path, relight=False, args=None):
        plydata = PlyData.read(path)
        s = self._texture_size

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        def read_grouped(prefix, channels):
            names = [p.name for p in plydata.elements[0].properties if p.name.startswith(prefix)]
            names = sorted(names, key=lambda x: int(x.split('_')[-1]))
            arr = np.zeros((xyz.shape[0], len(names)))
            for idx, attr_name in enumerate(names):
                arr[:, idx] = np.asarray(plydata.elements[0][attr_name])
            return arr.reshape((arr.shape[0], channels, s, s))

        texture_normal = read_grouped("tex_n", 2)
        metallic = read_grouped("metallic", 1)
        albedo = read_grouped("albedo", 3)
        roughness = read_grouped("roughness", 1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        self.active_sh_degree = self.max_sh_degree

        indirect_dc = np.zeros((xyz.shape[0], 3, 1))
        indirect_dc[:, 0, 0] = np.asarray(plydata.elements[0]["ind_dc_0"])
        indirect_dc[:, 1, 0] = np.asarray(plydata.elements[0]["ind_dc_1"])
        indirect_dc[:, 2, 0] = np.asarray(plydata.elements[0]["ind_dc_2"])

        extra_ind_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ind_rest_")]
        extra_ind_names = sorted(extra_ind_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_ind_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        indirect_extra = np.zeros((xyz.shape[0], len(extra_ind_names)))
        for idx, attr_name in enumerate(extra_ind_names):
            indirect_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        indirect_extra = indirect_extra.reshape((indirect_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if not relight:
            map_path1 = path.replace('.ply', '1.map')
            map_path2 = path.replace('.ply', '2.map')
            if os.path.exists(map_path1) and os.path.exists(map_path2):
                self.env_map = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res,
                                        min_roughness=args.envmap_min_roughness,
                                        max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map.load_state_dict(torch.load(map_path1))
                self.env_map.build_mips()
                self.env_map_2 = EnvLight(path=None, device='cuda', max_res=args.envmap_max_res,
                                          min_roughness=args.envmap_min_roughness,
                                          max_roughness=args.envmap_max_roughness, trainable=True).cuda()
                self.env_map_2.load_state_dict(torch.load(map_path2))
                self.env_map_2.build_mips()
        else:
            self.env_map = EnvLight(path=path.replace('.ply', '.hdr'), device='cuda', trainable=True).cuda()

        def as_param(a):
            return nn.Parameter(torch.tensor(a, dtype=torch.float, device="cuda").requires_grad_(True))

        self._xyz = as_param(xyz)
        self._texture_normal = as_param(texture_normal)
        self._metallic = as_param(metallic)
        self._roughness = as_param(roughness)
        self._albedo = as_param(albedo)
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_dc = nn.Parameter(torch.tensor(indirect_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._indirect_rest = nn.Parameter(torch.tensor(indirect_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = as_param(opacities)
        self._scaling = as_param(scales)
        self._rotation = as_param(rots)

    # ------------------------------------------------------------------ #
    # Densification is disabled during the texture stage.
    # ------------------------------------------------------------------ #
    def prune_points(self, mask):
        raise NotImplementedError("Densification is disabled during the texture stage")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        raise NotImplementedError("Densification is disabled during the texture stage")

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        raise NotImplementedError("Densification is disabled during the texture stage")

    def freeze_geometry(self):
        """Freeze positions for the texture stage."""
        self._xyz.requires_grad_(False)
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = 0.0
