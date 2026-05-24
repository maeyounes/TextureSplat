import torch
from scene import Scene
import os
import time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_surfel, render_surfel_textured
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene.gaussian_model import GaussianModel, TexturedGaussianModel
from plyfile import PlyData
from utils.image_utils import psnr
from utils.image_utils import compute_normal_mae_masked
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from torchvision.utils import save_image, make_grid
from PIL import Image


def load_gaussians_for_eval(model_path, iteration, sh_degree):
    """Pick GaussianModel vs TexturedGaussianModel by inspecting the saved ply,
    and return (model, render_fn). For textured plys the texture resolution is
    derived from the number of 'tex_n_*' attributes (2 * size * size)."""
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    props = [p.name for p in PlyData.read(ply_path).elements[0].properties]
    n_tex_n = sum(1 for p in props if p.startswith("tex_n_"))
    if n_tex_n > 0:
        texture_size = int(round((n_tex_n / 2) ** 0.5))
        print(f"Textured model detected (texture {texture_size}x{texture_size}).")
        return TexturedGaussianModel(sh_degree, texture_size=texture_size), render_surfel_textured
    print("Baseline model detected.")
    return GaussianModel(sh_degree), render_surfel


def render_set(model_path, views, gaussians, render_fn, pipeline, background, save_ims, opt, dataset : ModelParams, eval_normals : bool):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "test", "renders")
        color_path = os.path.join(render_path, 'rgb')
        normal_path = os.path.join(render_path, 'normal')
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)

    ssims = []
    psnrs = []
    lpipss = []
    render_times = []
    normal_maes = []

    gt_image_dir = os.path.join(dataset.source_path, "test")
    if not os.path.exists(gt_image_dir):
        print(f"Warning: Ground truth image directory not found at {gt_image_dir}. Skipping normal evaluation if enabled.")
        eval_normals = False # Disable if GT dir doesn't exist
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        # CUDA kernel launches are async, so we must sync before/after to
        # measure real GPU compute time instead of just kernel-launch overhead.
        torch.cuda.synchronize()
        t1 = time.time()

        rendering = render_fn(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        torch.cuda.synchronize()
        render_time = time.time() - t1
        
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(lpips(render_color, gt, net_type='vgg').item())
        render_times.append(render_time)
        
        


        if save_ims:
            # Save the rendered color image
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            # Save the normal map if available
            if 'rend_normal' in rendering:
                normal_map = rendering['rend_normal'] * 0.5 + 0.5
                rend_normal_filename = os.path.join(normal_path, '{0:05d}.png'.format(idx))
                torchvision.utils.save_image(normal_map, rend_normal_filename)
        
        
        # Evaluate normal metrics if flag is set and normals are rendered
        if eval_normals and 'rend_normal' in rendering and save_ims:
            gt_normal_filename = f"{view.image_name}_normal.png"
            gt_alpha_filename = f"{view.image_name}_alpha.png"
            gt_filename = f"{view.image_name}.png"
            gt_normal_path = os.path.join(gt_image_dir, gt_normal_filename)
            gt_alpha_path = os.path.join(gt_image_dir, gt_alpha_filename)
            gt_path = os.path.join(gt_image_dir, gt_filename)
            if os.path.exists(gt_normal_path):
                try:
                    try:
                        gt_normal_img = torchvision.io.read_image(gt_normal_path).float() / 255.0
                        # Load alpha mask (assuming it's compatible or handle similarly if needed)
                        gt_alpha_img = torchvision.io.read_image(gt_alpha_path).float().squeeze() / 255.0
                    except RuntimeError:
                        img = Image.open(gt_normal_path)
                        gt_normal_np = np.array(img)
                        # Ensure it's float and normalized
                        gt_normal_img = torch.from_numpy(gt_normal_np).float() / 255.0
                        gt_alpha_img = torch.from_numpy(np.array(Image.open(gt_path))).float()[..., 3] / 255.0
                        # Ensure CHW format
                        if len(gt_normal_img.shape) == 3 and gt_normal_img.shape[2] <= 4: # HWC -> CHW
                            gt_normal_img = gt_normal_img.permute(2, 0, 1)[:3, :, :]
                        else:
                            raise ValueError(f"Unexpected image shape from PIL for {gt_normal_path}: {gt_normal_img.shape}")

                    
                    # Convert GT normal from [0, 1] to [-1, 1]
                    rend_normal_filename = os.path.join(normal_path, '{0:05d}.png'.format(idx))
                    rend_normal_img = torchvision.io.read_image(rend_normal_filename).float() / 255.0
                    rend_normal_img = rend_normal_img * 2.0 - 1.0
                    gt_normal = (gt_normal_img[:3, :, :].to("cuda") * 2.0 - 1.0)
                    mae = compute_normal_mae_masked(rend_normal_img, gt_normal, gt_alpha_img)
                    normal_maes.append(mae)
                except Exception as e:
                    print(f"Warning: Could not load or process GT normal {gt_normal_path}: {e}")
            else:
                print(f"Warning: Ground truth normal file not found: {gt_normal_path}")     
    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()
    fps = 1.0 / np.array(render_times).mean()
    print('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))
        
    # Save normal metrics if evaluated
    if eval_normals and normal_maes:
        normal_mae_avg = np.mean(normal_maes)
        print("\nNormal Metrics Results:")
        print(f"Mean Angular Error (MAE) in degrees: {normal_mae_avg:.4f}")

        normal_metrics_path = os.path.join(model_path, "normal_evaluation.txt")
        with open(normal_metrics_path, 'w') as f:
            f.write(f"normal_mae: {normal_mae_avg}\n")
    elif eval_normals:
        print("\nNormal evaluation enabled, but no valid normals were compared (check GT paths and rendered output).")



   
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op, indirect, eval_normals=False):
    with torch.no_grad():
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        gaussians, render_fn = load_gaussians_for_eval(dataset.model_path, iteration, dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if indirect:
            op.indirect = 1
            gaussians.load_mesh_from_ply(dataset.model_path, iteration)


        render_set(dataset.model_path, scene.getTestCameras(), gaussians, render_fn, pipeline, background, save_ims, op, dataset, eval_normals)
        
        env_dict = gaussians.render_env_map()
        grid = [
            env_dict["env1"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env1.png"))
        grid = [
            env_dict["env2"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env2.png"))
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--eval_normals", action="store_true", help="Evaluate normal maps using Mean Angular Error (requires GT normals)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images, op, True, args.eval_normals)
