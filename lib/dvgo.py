import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch_scatter import segment_coo
from . import grid
import numpy as np
parent_dir = os.path.dirname(os.path.abspath(__file__))

from dvgo_cu import render_utils_cuda

def create_pseudo_label_v1(ray_id, step_id, depth, t_min, interval_dist):
    """
    ray_id:(Np,)            the ray_id of each point
    depth:(8192,)           the depth of each ray
    backward_depth:(8192,)  the backward depth of each ray
    step_id:(Np,)           the step_id of each point
    t_min:(8192,)           the t_min of each ray
    interval_dist:          the interval dist between voxels
    
    return:
    label:(Np,)             the label value of each ray
    """
    Np = step_id.shape[0]
    if Np == 0:
        return
    # compute step_id diff between rendered depth and t_min
    depth_id = ((depth - t_min) / interval_dist).int() # 8192, some of which is negative
    depth_label = depth_id[ray_id] # Np, the depth_id of each points on corresponding ray, some which is negative
    
    ## find the depth points
    mask = depth_label == step_id # Np, if is the depth points
    
    return mask.float() ## (N_points,)

'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 # additional/change
                 fast_color_thres_init=0,fast_color_thres_final=0,
                 N_iters=0,
                 N_dynamic_iters=0,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres_init = fast_color_thres_init
        self.fast_color_thres_final = fast_color_thres_final
        self.fast_color_thres = fast_color_thres_init
        self.N_iters = N_iters
        self.N_dynamic_iters = N_dynamic_iters

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density_factor = 1
        if "density_factor" in kwargs:
            self.density_factor=kwargs['density_factor']

        self.density = grid.create_grid(
                density_type, channels=1, world_size=self.world_size, #[edge*self.density_factor for edge in self.world_size],
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dvgo: feature voxel grid', self.k0)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        self.importance = None
        self.used_kwargs = {'density_factor':self.density_factor}
        
        print('initialization finished')
        
    def cut_half(self):
        ori_shape = self.mask_cache.mask.shape
        print("mask size:", ori_shape)
        print("Cut weight")
        
        self.mask_cache.mask[:, :ori_shape[1]//2, :] = 0

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'fast_color_thres_init': self.fast_color_thres_init,
            'fast_color_thres_final': self.fast_color_thres_final,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            **self.rgbnet_kwargs,
            **self.used_kwargs,
        }

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density.grid[nearest_dist[None,None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid([self.world_size[0]*self.density_factor,self.world_size[1]*self.density_factor,self.world_size[2]*self.density_factor])
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]*self.density_factor),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]*self.density_factor),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]*self.density_factor),
            ), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                    path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self, global_step, cur_thres=1):
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        ## update color_thres
        if global_step == -1:
            self.fast_color_thres = self.fast_color_thres_final
            self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
           
        elif self.N_iters > 0:
            self.fast_color_thres = self.fast_color_thres_init * (self.N_iters-global_step)/self.N_iters + self.fast_color_thres_final * global_step / self.N_iters
            self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        if global_step >= self.N_dynamic_iters and cur_thres != 1.0:
            importance = self.importance.flatten()  
            # dynamic pruning
            if global_step == self.N_dynamic_iters:
                percent_sum = cur_thres
            else:
                percent_sum = 1 - (global_step - self.N_dynamic_iters) / (self.N_iters - self.N_dynamic_iters) * (1 - cur_thres)
            
            vals,idx = sorted_importance = torch.sort(importance+(1e-6))
            cumsum_val = torch.cumsum(vals, dim=0)
            split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
            split_val_nonprune = vals[split_index]
            percent_point = (importance+(1e-6)>= vals[split_index]).sum()/importance.numel()
            print(f'{percent_point*100:.2f}% of most important points contribute over {(percent_sum)*100:.2f}% importance ')
            self.non_prune_mask = importance>split_val_nonprune ## False been prune
            print("original number of voxels:", torch.sum(self.mask_cache.mask))
            self.mask_cache.mask &= self.non_prune_mask.reshape(self.importance.shape[-3:])
            print("number of importance voxels:", torch.sum(self.non_prune_mask))
            print("changed number of voxels:", torch.sum(self.mask_cache.mask))

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu())+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += (ones.grid.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, t_min, t_max, N_steps, stepdist

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, target=None, use_vq_flag=None, include_thres=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, t_min, t_max, N_steps, interval_dist = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            
        debug = False
        sampled_importance = None
        with utils.Timing('-density sampling', debug):
            # query for alpha w/ post-activation
            if self.importance is not None:
                density, sampled_importance = self.density(ray_pts, importance=self.importance)
            
            else:
                density = self.density(ray_pts, importance=None)

        with utils.Timing('-density to alpha', debug):
            alpha = self.activate_density(density, interval)
            if self.fast_color_thres > 0:
                mask = (alpha > self.fast_color_thres)
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                density = density[mask]
                alpha = alpha[mask]
                if sampled_importance is not None:
                    sampled_importance = sampled_importance[mask]

        # compute accumulated transmittance
        with utils.Timing('-alpha to weight', debug):   
            surf_mask = None
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
            if render_kwargs.get('depth_label', False) and (
                global_step > render_kwargs['entropy_loss_after'] and 
                global_step <= render_kwargs['entropy_loss_before'] and 
                global_step%render_kwargs['depth_entropy_every']==0
            ):
                depth = segment_coo(
                        src=(weights * (step_id * interval_dist + t_min[ray_id])),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
                depth_label = create_pseudo_label_v1(ray_id, step_id, depth, t_min, interval_dist)
                
            if self.fast_color_thres > 0:
                mask = (weights > self.fast_color_thres)
                weights = weights[mask]
                alpha = alpha[mask]
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                density = density[mask]
                if sampled_importance is not None:
                    sampled_importance = sampled_importance[mask]
                     
            if render_kwargs.get('depth_label', False) and (
                global_step > render_kwargs['entropy_loss_after'] and 
                global_step <= render_kwargs['entropy_loss_before'] and 
                global_step % render_kwargs['depth_entropy_every']==0
                ):
                if self.fast_color_thres > 0:
                    depth_label =  depth_label[mask]
                if render_kwargs['weight_surface_distill'] > 0:
                    surf_mask = depth_label.bool()
                        
                ret_dict.update({'depth_label': depth_label})

        with utils.Timing('-feat query', debug):
            if self.rgbnet_full_implicit:
                pass
            else:
                k0 = self.k0(ray_pts)
            if self.rgbnet is None:
                # no view-depend effect
                rgb = torch.sigmoid(k0)
            else:
                # view-dependent color emission
                if self.rgbnet_direct:
                    k0_view = k0
                else:
                    k0_view = k0[:, 3:]
                    k0_diffuse = k0[:, :3]
                with utils.Timing('-rgb net - pre', debug):
                    viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
                    viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
                    viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
                    rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
                  
                with utils.Timing('-rgb net - forward', debug):
                    rgb_logit = self.rgbnet(rgb_feat)
                    if self.rgbnet_direct:
                        rgb = torch.sigmoid(rgb_logit)
                    else:
                        rgb = torch.sigmoid(rgb_logit + k0_diffuse)
                        
                    ## surface color
                    if surf_mask is not None:
                        surf_rgb = rgb[surf_mask]
                        surf_weights = weights[surf_mask]
                        surf_ray_id = ray_id[surf_mask]
                        surf_marched = segment_coo(
                            src=(surf_weights.unsqueeze(-1) * surf_rgb),
                            index=surf_ray_id,
                            out=torch.zeros([N, 3]),
                            reduce='sum')
                        surf_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
                        ret_dict.update({
                            'surf_marched': surf_marched,
                        })
                        
        with utils.Timing('-ray march', debug):
            # Ray marching
            rgb_marched = segment_coo(
                    src=(weights.unsqueeze(-1) * rgb),
                    index=ray_id,
                    out=torch.zeros([N, 3]),
                    reduce='sum')  

        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'density':density,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * step_id),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict

    def forward_imp(self, rays_o, rays_d, viewdirs, pseudo_grid, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, t_min, t_max, N_steps, stepdist = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
        debug = False
        with utils.Timing('-alpha calc', debug):
            # query for alpha w/ post-activation
           
            density, sampled_pseudo_grid = self.density(ray_pts, importance=pseudo_grid)

            alpha = self.activate_density(density, interval)
            if self.fast_color_thres > 0:
                mask = (alpha > self.fast_color_thres)
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                density = density[mask]
                alpha = alpha[mask]
                sampled_pseudo_grid = sampled_pseudo_grid[mask]
           
            # compute accumulated transmittance
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
            if self.fast_color_thres > 0:
                mask = (weights > self.fast_color_thres)
                weights = weights[mask]
                alpha = alpha[mask]
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                step_id = step_id[mask]
                density = density[mask]
                sampled_pseudo_grid = sampled_pseudo_grid[mask]

        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'raw_alpha': alpha,
            'ray_id': ray_id,
            'density':density,
            'sampled_pseudo_grid':sampled_pseudo_grid
        })
        return ret_dict

    @torch.no_grad()
    def init_cdf_mask(self, thres=1.0):
        print("start cdf three split")
        importance = self.importance.flatten()   
        if thres!=1.0:
            percent_sum = thres
            vals,idx = sorted_importance = torch.sort(importance+(1e-6))
            cumsum_val = torch.cumsum(vals, dim=0)
            split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
            split_val_nonprune = vals[split_index]
            percent_point = (importance+(1e-6)>= vals[split_index]).sum()/importance.numel()
            print(f'{percent_point*100:.2f}% of most important points contribute over {(percent_sum)*100:.2f}% importance ')
            self.non_prune_mask = importance>split_val_nonprune 
            self.non_prune_mask &= self.mask_cache.mask.flatten()
        else: 
            self.non_prune_mask = torch.ones_like(importance).bool()
            self.non_prune_mask &= self.mask_cache.mask.flatten()

        return self.non_prune_mask

    @torch.no_grad()
    def quantize_reformat(self, thres=1.0, quantize=False, save_path=None):
       
        print("start tensor quantization")
        k0_grid = self.k0.grid.reshape(self.k0_dim,-1)
        k0_grid = k0_grid.T

        density_grid = self.density.grid.reshape(1,-1)
        density_grid = density_grid.T

        self.init_cdf_mask(thres=thres)

        new_k0_grid = torch.zeros_like(k0_grid)
        new_densiy_grid = torch.zeros_like(density_grid)# - 99999
        
        non_prune_density = density_grid[self.non_prune_mask,:]
        if quantize:
            non_prune_density = torch.quantize_per_tensor(non_prune_density, scale=non_prune_density.std()/15, zero_point=torch.round(non_prune_density.mean()), dtype=torch.qint8)
            new_densiy_grid[self.non_prune_mask,:] = non_prune_density.dequantize() 
        else:
            new_densiy_grid[self.non_prune_mask,:] = non_prune_density 
        # new_k0_grid[self.non_prune_mask,:] = k0_grid[self.non_prune_mask,:]
        non_prune_grid = k0_grid[self.non_prune_mask,:]
        if quantize:
            non_prune_grid = torch.quantize_per_tensor(non_prune_grid, scale=non_prune_grid.std()/15, zero_point=torch.round(non_prune_grid.mean()), dtype=torch.qint8)
            new_k0_grid[self.non_prune_mask,:] =  non_prune_grid.dequantize() 
        else:
            new_k0_grid[self.non_prune_mask,:] =  non_prune_grid
            
        if save_path is not None:
            import numpy as np
            import math
            from copy import deepcopy
            os.makedirs(f'{save_path}/extreme_saving', exist_ok=True)
            np.savez_compressed(f'{save_path}/extreme_saving/non_prune_density.npz',non_prune_density.int_repr().cpu().numpy())
            np.savez_compressed(f'{save_path}/extreme_saving/non_prune_grid.npz',non_prune_grid.int_repr().cpu().numpy())
            np.savez_compressed(f'{save_path}/extreme_saving/non_prune_mask.npz',np.packbits(self.non_prune_mask.reshape(-1).cpu().numpy()))
            np.savez_compressed(f'{save_path}/extreme_saving/rgbnet.npz',deepcopy(self.rgbnet).half().cpu().state_dict())

            # we also save necessary metadata 
            metadata = dict()
            metadata['global_step'] =20000
            metadata['world_size'] = self.world_size
            metadata['model_kwargs'] = self.get_kwargs()
            metadata['model_state_dict'] = dict()
            metadata['grid_dequant'] = dict()
            metadata['grid_dequant']['zero_point'] = non_prune_grid.q_zero_point()
            metadata['grid_dequant']['scale'] = non_prune_grid.q_scale()
            metadata['density_dequant'] = dict()
            metadata['density_dequant']['zero_point'] = non_prune_density.q_zero_point()
            metadata['density_dequant']['scale'] = non_prune_density.q_scale()
            model_state_dict = self.state_dict()
            metadata['model_state_dict']['act_shift'] = model_state_dict['act_shift']
            metadata['model_state_dict']['viewfreq'] = model_state_dict['viewfreq']
            metadata['model_state_dict']['xyz_min'] = model_state_dict['xyz_min']
            metadata['model_state_dict']['xyz_max'] = model_state_dict['xyz_max']
            metadata['model_state_dict']['density.xyz_min'] = model_state_dict['density.xyz_min']
            metadata['model_state_dict']['density.xyz_max'] = model_state_dict['density.xyz_max']
            metadata['model_state_dict']['k0.xyz_min'] = model_state_dict['k0.xyz_min']
            metadata['model_state_dict']['k0.xyz_max'] = model_state_dict['k0.xyz_max']
            np.savez_compressed(f'{save_path}/extreme_saving/metadata.npz', metadata=metadata)

            # zip everything together to get final size
            os.system(f"zip -r {save_path}/extreme_saving.zip {save_path}/extreme_saving ")

        new_k0_grid = new_k0_grid.T.reshape(*self.k0.grid.shape).contiguous()
        new_densiy_grid = new_densiy_grid.T.reshape(*self.density.grid.shape).contiguous()
        self.k0.grid = torch.nn.Parameter(new_k0_grid)
        self.density.grid = torch.nn.Parameter(new_densiy_grid)
       
        print("finish fully vector quantize")

''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

class Raw2Alpha_nonuni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        exp, alpha = render_utils_cuda.raw2alpha_nonuni(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_nonuni_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

