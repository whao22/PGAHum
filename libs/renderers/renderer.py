import torch
import torch.nn.functional as F


import pytorch3d.ops as ops
from libs.embeders.hannw_fourier import get_embedder
from libs.utils.FastMinv.fast_matrix_inv import FastDiff4x4MinvFunction
from libs.utils.general_utils import normalize_canonical_points, hierarchical_softmax, sample_sdf, sample_sdf_from_grid
from libs.utils.geometry_utils import compute_gradient

SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}

class IDHRenderer:
    def __init__(self,
                 pose_decoder,
                 motion_basis_computer,
                 offset_net,
                 skinning_model,
                 non_rigid_mlp,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 is_adasamp,
                 total_bones,
                 sdf_mode,
                 inner_sampling,
                 non_rigid_multries,
                 N_iter_backward,
                 non_rigid_kick_in_iter,
                 non_rigid_full_band_iter,
                 pose_refine_kick_in_iter,
                 use_init_sdf,
                 sdf_threshold):
        self.pose_decoder = pose_decoder
        self.motion_basis_computer = motion_basis_computer
        self.offset_net = offset_net
        self.skinning_model = skinning_model
        self.non_rigid_mlp = non_rigid_mlp
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.is_adasamp = is_adasamp
        self.total_bones = total_bones
        self.sdf_mode = sdf_mode
        self.inner_sampling = inner_sampling
        self.non_rigid_multries = non_rigid_multries
        self.N_iter_backward = N_iter_backward
        self.use_init_sdf = use_init_sdf
        self.non_rigid_kick_in_iter = non_rigid_kick_in_iter
        self.non_rigid_full_band_iter = non_rigid_full_band_iter
        self.pose_refine_kick_in_iter = pose_refine_kick_in_iter
        self.sdf_threshold = sdf_threshold
    
    def dirs_from_pts(self, pts):
        """calculate direction of points by next points minus current points

        Args:
            pts (Tensor): (N_rays, N_samples, 3)

        Returns:
            Tensor: (N_rays, N_samples, 3)
        """
        dirs = torch.zeros_like(pts)
        dirs[:, :-1] = pts[:, 1:]-pts[:, :-1]
        dirs[:, -1] = dirs[:, -2]
        
        norm = torch.norm(dirs, p=2, dim=-1, keepdim=True)
        dirs = dirs / (norm + 1e-12)
        return dirs

    def deform_dirs(self, ray_dirs, points_cnl, transforms_bwd): ## TODO confirm whether the implement is right or not.
        N_rays, N_samples, _ = points_cnl.shape
        ray_dirs = torch.tile(ray_dirs[:, None, :], [1, N_samples, 1])
        dirs_cnl = torch.matmul(transforms_bwd[..., :3, :3].squeeze(), -ray_dirs.view(-1, 3, 1))
        return dirs_cnl.reshape(N_rays, N_samples, 3)
        
    def render_outside(self, rays_o, rays_d, z_vals, deform_kwargs, background_rgb=None):
        """Render background """
        N_rays, N_samples = z_vals.shape

        # Section length
        dists = torch.zeros_like(z_vals)
        dists[..., :-1] = z_vals[..., 1:] - z_vals[..., :-1]
        dists[..., -1:] = dists[..., :-1].mean(dim=-1, keepdim=True)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts_obs = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # (N_rays, N_samples, 3)
        pts_cnl, _, _, _, _ = self.deform_points(pts_obs, **deform_kwargs) # (N_rays, N_samples, 3)
        # pts_cnl, pts_mask = self.deform_points2(pts_obs, **deform_kwargs)
        dirs = self.dirs_from_pts(pts_cnl) # (N_rays, N_samples, 3)
        dirs = dirs.reshape(-1, 3) # (N_rays x N_samples, 3)

        dis_to_center = torch.linalg.norm(pts_cnl, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts_cnl = torch.cat([pts_cnl / dis_to_center, 1.0 / dis_to_center], dim=-1)       # (N_rays x N_samples, 4)
        pts_cnl = pts_cnl.reshape(-1, 3 + int(self.n_outside > 0)) # (batch_size x N_samples, 4)
        
        density, sampled_color = self.nerf(pts_cnl, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(N_rays, N_samples)) * dists)
        alpha = alpha.reshape(N_rays, N_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([N_rays, 1]).to(alpha), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(N_rays, N_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }
    
    def render_core(self, 
                    rays_o,
                    rays_d,
                    z_vals,
                    sdf_decoder,
                    deform_kwargs,
                    sdf_kwargs,
                    background_color,
                    background_alpha,
                    background_sampled_color,
                    cos_anneal_ratio):
        """render rgb and normal from sdf

        Args:
            points_obs (Tensor): (batch_size, N_rays, N_samples, 3)
            points_cnl (Tensor): (batch_size, N_rays, N_samples, 3)
            t (Tensor): (batch_size, N_rays, N_samples, 1)
            cos_anneal_ratio (float): _description_

        Returns:
            _type_: _description_
        """
        N_rays, N_samples = z_vals.shape
        device = z_vals.device
        
        # Section length
        dists = torch.zeros_like(z_vals)
        dists[..., :-1] = z_vals[..., 1:] - z_vals[..., :-1]
        dists[..., -1:] = dists[..., :-1].mean(-1, keepdim=True)
        mid_z_vals = z_vals + dists * 0.5
        mid_dists = torch.zeros_like(mid_z_vals)
        mid_dists[..., :-1] = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]
        mid_dists[..., -1:] = mid_dists[..., :-1].mean(-1, keepdim=True)
        
        # section midpoints
        points_obs = rays_o[..., None, :] + rays_d[..., None, :] * mid_z_vals[..., None]  # (N_rays, N_samples, 3)
        points_cnl, pts_W_pred, pts_W_sampled, _, _ = self.deform_points(points_obs, **deform_kwargs)  # (N_rays, N_samples, 3)
        # pts_cnl, pts_mask = self.deform_points2(points_obs, **deform_kwargs)  # (N_rays, N_samples, 3)
        
        dirs = self.dirs_from_pts(points_cnl) # (N_rays, N_samples, 3)
        # dirs = self.deform_dirs(rays_d, points_cnl, transforms_bwd) # (N_rays, N_samples, 3)
        points_cnl = points_cnl.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        
        s = self.deviation_network(torch.zeros([1]).to(points_cnl)).clip(1e-6, 1e3).detach()
        
        # nn forward
        if self.sdf_mode == 'hyper_net':
            feature_vector, sdf = sample_sdf(points_cnl, sdf_decoder, out_feature=True, **sdf_kwargs)
            gradients = compute_gradient(sdf, points_cnl)
        elif self.sdf_mode in ['mlp','tri']:
            sdf_nn_output = self.sdf_network(points_cnl)
            delta_sdf = sdf_nn_output[:, :1]
            feature_vector = sdf_nn_output[:, 1:]
            if self.use_init_sdf:
                init_sdf = sample_sdf_from_grid(points_cnl, **sdf_kwargs)
                sdf = init_sdf + delta_sdf
                distance_w_ = init_sdf.clone().detach() - self.sdf_threshold
                distance_w = distance_w_.clone()
                distance_w[distance_w_<=0] = 1.
                distance_w[distance_w_> 0] = 0.
                distance_w = distance_w.reshape(N_rays, N_samples) # TODO 这个分段函数可能效果不明显
            else:
                sdf = delta_sdf
                distance_w = pts_W_sampled.clone().detach().sum(dim=-1).reshape(N_rays, N_samples)
            gradients = compute_gradient(sdf, points_cnl)
        else:
            raise ValueError(f'The sdf_mode is {self.sdf_mode}, which is not valid.')
        
        sampled_color = self.color_network(points_cnl, gradients, dirs, feature_vector).reshape(N_rays, N_samples, 3)
        
        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        inv_s = self.deviation_network(torch.zeros([1, 3]).to(points_cnl))[:, :1].clip(1e-6, 1e6)           # Single parameter

        sigmoid_sdf = torch.sigmoid(s * sdf)
        weight_sdf = s * sigmoid_sdf * (1 - sigmoid_sdf)
        weight_sdf = weight_sdf.view(N_rays, N_samples, 1) / (weight_sdf.view(N_rays, N_samples, 1).sum(dim=1, keepdims=True) + 1e-6)
        weight_sdf[weight_sdf.sum(1).squeeze() < 0.2] = torch.ones([N_samples]).to(points_cnl).unsqueeze(1) / N_samples
        inv_s = (inv_s.expand(N_rays, N_samples, 1) *
                 torch.exp((gradients.norm(dim=1, keepdim=True).view(N_rays, N_samples, 1) * weight_sdf.detach())
                           .sum(dim=1, keepdim=True) - 1)).view(N_rays * N_samples, 1)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # sigma
        cdf = torch.sigmoid(sdf * inv_s)
        e = inv_s * (1 - cdf) * (-iter_cos) * mid_dists.reshape(-1, 1)
        alpha = (1 - torch.exp(-e)).reshape(N_rays, N_samples).clip(0.0, 1.0)
        
        # Render with background
        if background_alpha is not None:
            pts_norm = torch.linalg.norm(points_cnl, ord=2, dim=-1, keepdim=True).reshape(N_rays, N_samples)
            inside_sphere = (pts_norm < 1.0).float().detach()
            relax_inside_sphere = (pts_norm < 1.2).float().detach()
            alpha = alpha * inside_sphere + background_alpha[:, :N_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, N_samples:]], dim=-1)
            
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :N_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, N_samples:]], dim=1)
        alpha = alpha * distance_w
        weights = alpha * torch.cumprod(torch.cat([torch.ones([N_rays, 1]).to(points_cnl), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_color is not None:    # Fixed background, usually black
            color = color + background_color * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(N_rays, N_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        # gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        gradient_error = gradient_error.sum() / (gradient_error.numel() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(N_rays, N_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': cdf.reshape(N_rays, N_samples),
            'gradient_error': gradient_error,
            # 'inside_sphere': inside_sphere,
            'pts_W_pred': pts_W_pred,
            "pts_W_sampled": pts_W_sampled
        }
    
    def render(self, data, iter_step, cos_anneal_ratio=0.0, sdf_decoder=None):
        # unpack data
        ## rays
        batch_rays = data['batch_rays'].squeeze() # (N_rays, 12)
        rays_o = batch_rays[..., 0:3] # (N_rays, 3)
        rays_d = batch_rays[..., 3:6] # (N_rays, 3)
        if self.inner_sampling:
            z_vals = data['z_vals'].squeeze() # (N_rays, N_samples)
            near, far = z_vals[:, :1], z_vals[:, -1:]
        else:
            near = batch_rays[..., 10:11] # (N_rays, 3)
            far = batch_rays[..., 11:12] # (N_rays, 3)
            
        ## motion
        tjoints = data['tjoints'] # (batch_size, 24, 3)
        gtfs_02v = data['gtfs_02v'] # (batch_size, 24, 4, 4)
        dst_Rs = data['dst_Rs'] # (batch_size, 24, 3, 3)
        dst_Ts = data['dst_Ts'] # (batch_size, 24, 3)
        dst_posevec = data['dst_posevec'] # (batch_size, 69)
        dst_vertices = data['dst_vertices'] # (batch_size, 6890, 3)
        cnl_gtfms = data['cnl_gtfms'] # (batch_size, 24, 4, 4)
        skinning_weights = data['skinning_weights'] # (batch_size, 6890, 24)
        cnl_bbmin = data['smpl_sdf']['bbmin']
        cnl_bbmax = data['smpl_sdf']['bbmax']
        cnl_grid_sdf = data['smpl_sdf']['sdf_grid']
        ## render
        background_color = data['background_color'][0] # (3)
        
        N_rays, _ = batch_rays.shape
        if self.inner_sampling:
            N_samples = z_vals.size(-1)
        else:
            N_samples = self.n_samples
        N_outside = self.n_outside

        # dst_gtfms (batch_size, 24, 4, 4)
        dst_gtfms, pose_refine_error = self.get_dst_gtfms(iter_step, self.pose_refine_kick_in_iter, dst_Rs, dst_Ts, dst_posevec, self.total_bones, cnl_gtfms, tjoints, gtfs_02v)
        
        # get_non_rigid_embeder
        non_rigid_pos_embed_fn, _ = get_embedder(multires=self.non_rigid_multries, 
                                                 iter_val=iter_step,
                                                 full_band_iter=self.non_rigid_full_band_iter,
                                                 kick_in_iter=self.non_rigid_kick_in_iter)
        
        # pack required data which used in deforming points
        sdf_kwargs={
            "cnl_bbmin": cnl_bbmin,
            "cnl_bbmax": cnl_bbmax,
            "cnl_grid_sdf": cnl_grid_sdf,
        }
        deform_kwargs={
            "skinning_weights": skinning_weights,
            "dst_gtfms": dst_gtfms,
            "dst_posevec": dst_posevec,
            "dst_vertices": dst_vertices,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "iter_step": iter_step,
            "non_rigid_kick_in_iter": self.non_rigid_kick_in_iter,
        }
        
        if False:
            import pickle
            import trimesh
            
            with open("data/data_prepared/CoreView_377/0/canonical_joints.pkl","rb") as f: 
                data = pickle.load(f)
            tpose_vertices = data['vertices']
            tpose_vertices= torch.from_numpy(tpose_vertices).to(gtfs_02v).float()[None, ...]
            
            transforms_fwda = torch.matmul(skinning_weights, torch.inverse(gtfs_02v).view(1, -1, 16)).view(1, 6890, 4, 4)
            transforms_bwd = torch.inverse(transforms_fwda)

            homogen_coord = torch.ones(1, 6890, 1, dtype=torch.float32, device=dst_gtfms.device)
            points_homo = torch.cat([tpose_vertices, homogen_coord], dim=-1).view(1, 6890, 4, 1)
            points_newa = torch.matmul(transforms_bwd, points_homo)[:, :, :3, 0]
            trimesh.Trimesh(points_newa[0].detach().cpu().numpy()).export("tvpose_vertices_from_tpose.obj")
            
        if self.inner_sampling:
            if False:
                # additionally sample more points in rays
                N_extra = N_samples // 8
                z_vals_dist = (z_vals[:,-1:] - z_vals[:, :1])/(N_samples-1) # (N_rays, 1)
                extra_z_vals_near = torch.tile(-z_vals_dist, [1, N_extra])
                extra_z_vals_near = torch.cumsum(extra_z_vals_near, dim=-1).flip(dims=[-1]) + z_vals[:, :1] # (N_rays, N_extra)
                extra_z_vals_far = torch.tile(z_vals_dist, [1, N_extra])
                extra_z_vals_far = torch.cumsum(extra_z_vals_far, dim=-1) + z_vals[:, -1:] # (N_rays, N_extra)
                z_vals = torch.concat([extra_z_vals_near, z_vals, extra_z_vals_far], dim=-1) # (N_rays, N_samples + 2*N_extra)
                self.n_samples = N_samples + 2 * N_extra
                N_samples = self.n_samples
        else:
            # calculate depth in rays
            sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
            z_vals = torch.linspace(0.0, 1.0, N_samples).to(near)
            z_vals = near + (far - near) * z_vals[None, :] # (batch_size, n_samples)
        
        if N_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (N_outside + 1.0), N_outside).to(near) # (N_outside, )
        
        if self.perturb > 0:
            # t_rand = torch.rand_like(z_vals) - 0.5
            # z_dists = torch.zeros_like(z_vals)
            # z_dists[..., :-1] = z_vals[..., 1:] - z_vals[..., :-1]
            # z_dists[..., -1:] = z_dists[..., :-1].mean(dim=-1, keepdim=True)
            # z_vals = z_vals + t_rand * z_dists # (N_rays, N_samples)
            t_rand = (torch.rand([N_rays, 1]).to(near) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / N_samples # (batch_size, n_samples)
            
            if N_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([N_rays, N_outside]).to(near)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand # (N_rays, N_outside)
        
        if N_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / N_samples # (N_rays, N_outside)
        
        # Background model
        background_alpha = None
        background_sampled_color = None
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1) # (batch_size, n_sample+n_outside)
            ret_outside = self.render_outside(rays_o, rays_d, z_vals_feed, deform_kwargs)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']
        
        # render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sdf_decoder,
                                    deform_kwargs,
                                    sdf_kwargs,
                                    background_color=background_color,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)
        
        ret_fine['pose_refine_error'] = pose_refine_error
        return ret_fine
    
    def get_dst_gtfms(self, iter_step, pose_refine_kick_in_iter, dst_Rs, dst_Ts, dst_posevec, total_bones, cnl_gtfms, tjoints, gtfs_02v):
        pose_refine_error = None
        # pose refine and motion basis calculation
        if iter_step >= pose_refine_kick_in_iter:
            dst_Rs, dst_Ts, pose_refine_error = self.pose_refine(dst_Rs, dst_Ts, dst_posevec, total_bones)
        dst_gtfms = self.motion_basis_computer(dst_Rs, dst_Ts, cnl_gtfms, tjoints)
        gtfs_02v_inv = torch.inverse(gtfs_02v)
        # gtfs_02v_inv = FastDiff4x4MinvFunction.apply(gtfs_02v.reshape(-1, 4, 4))[0].reshape_as(gtfs_02v)
        dst_gtfms = torch.matmul(dst_gtfms, gtfs_02v_inv) # final bone transforms that transforms the canonical
                                                                    # Vitruvian-pose mesh to the posed mesh, without global
                                                                    # translation
        return dst_gtfms, pose_refine_error
    
    def multiply_corrected_Rs(self, Rs, correct_Rs, total_bones):
        total_bones = total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)
    
    def multiply_corrected_Rs_hirarch(self, Rs, correct_Rs, total_bones):
        """_summary_

        Args:
            Rs (Tensor): shape (B, 24, 3, 3)
            correct_Rs (Tensor): shape (B, 23, 3, 3)
            total_bones (int): 24
        """
        # total_bones = total_bones - 1
        Rs_ret = Rs.clone()
        for i in range(1, total_bones): # TODO loop unwinding to speedup
            Rs_ret[:, i, ...] = torch.matmul(Rs_ret[:, SMPL_PARENT[i], ...].clone(), correct_Rs[:, i-1, ...])
        
        return Rs_ret
    
    def pose_refine(self, dst_Rs, dst_Ts, dst_posevec, total_bones):
        # forward pose refine
        pose_out = self.pose_decoder(dst_posevec)
        refined_Rs = pose_out['Rs'] # (B, 23, 3, 3) Rs matrix
        refined_Ts = pose_out.get('Ts', None)
        
        # correct dst_Rs
        # ignore_refine_Rs = False
        if True:
            dst_Rs_no_root = dst_Rs[:, 1:, ...] # (B, 23, 3, 3)
            dst_Rs_no_root = self.multiply_corrected_Rs(dst_Rs_no_root, refined_Rs, total_bones) # (B, 23, 3, 3)
            dst_Rs = torch.cat([dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1) # (B, 24, 3, 3)
        
        if False:
            dst_Rs = self.multiply_corrected_Rs_hirarch(dst_Rs, refined_Rs, total_bones)
            
        # correct dst_Ts
        if refined_Ts is not None:
            dst_Ts = dst_Ts + refined_Ts
        
        # pose_refine_error = None
        # pose_refine_error = torch.norm(refined_Rs - torch.eye(3,3).to(refined_Rs), p=2)
        pose_refine_error = torch.nn.functional.l1_loss(refined_Rs, torch.eye(3,3).expand_as(refined_Rs).to(refined_Rs))
        return dst_Rs, dst_Ts, pose_refine_error
    
    def deform_points(self, points, dst_posevec=None, iter_step=1, non_rigid_kick_in_iter=1000, dst_gtfms=None, non_rigid_pos_embed_fn=None, **deform_kwargs):
        """deform the points in observation space into cononical space via Itertative Backward Deformation.

        Args:
            points (tensor): (N, 3)
            dst_posevec (tensor, optional): pose vector. Defaults to None.
            iter_step (int, optional): _description_. Defaults to 1.
            non_rigid_kick_in_iter (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        N_rays, N_samples, _ = points.shape
        points = points.reshape(1, -1, 3) # # (1, N_points, 3)
        points_skl, transforms_fwd, transforms_bwd, pts_W_sampled = self.backward_lbs_knn(points.clone(), dst_gtfms.clone(), **deform_kwargs) # (1, N_points, 3)
        
        pts_W_pred = None
        # if N_iter_backward >= 1, use iterative backward deformation, cnl = points_obs * weights
        if self.N_iter_backward > 0 and iter_step >= non_rigid_kick_in_iter:
            for i in range(self.N_iter_backward):
                points_skl, transforms_fwd, transforms_bwd, pts_W_pred = self.backward_lbs_nn(points_skl.clone(), points.clone(), dst_gtfms.clone()) # (1, N_points, 3)
            points_cnl = points_skl
        
        # if N_iter_backward <= 0, not use iterative backward deformation, cnl = points_skl + points_offset
        elif self.N_iter_backward <= 0:
            points_skl = points_skl.reshape(-1, 3) # (N_points, 3)
            non_rigid_embed_xyz = non_rigid_pos_embed_fn(points_skl)
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) if iter_step < non_rigid_kick_in_iter else dst_posevec # (1, 69)
            points_cnl = self.non_rigid_mlp(pos_embed=non_rigid_embed_xyz, pos_xyz=points_skl, condition_code=non_rigid_mlp_input)['xyz']
        
        return points_cnl.reshape(N_rays, N_samples, 3), pts_W_pred, pts_W_sampled, transforms_fwd, transforms_bwd,

    def query_weights(self, points):
        """Canonical point -> deformed point

        Args:
            points (_type_): (N, 3)

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        wi = self.skinning_model(points, c=torch.empty(points.size(0), 0, device=points.device, dtype=torch.float32))

        if wi.size(-1) == 24:
            w_ret = F.softmax(wi, dim=-1) # naive softmax
        elif wi.size(-1) == 25:
            w_ret = hierarchical_softmax(wi * 20) # hierarchical softmax in SNARF
        else:
            raise ValueError('Wrong output size of skinning network. Expected 24 or 25, got {}.'.format(wi.size(-1)))

        return w_ret

    def backward_lbs_nn(self, points_cnl, points, dst_gtfms, inverse=False):
        ''' Backward skinning based on neural network predicted skinning weights 
        Args:
            points (tensor): canonical points. shape: [B, N, D]
            pts_weights (tensor): conditional input. [B, N, J]
            dst_gtfms (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            x (tensor): skinned points. shape: [B, N, D]
        '''
        batch_size, N_points, _ = points.shape
        device = points.device
        
        pts_weights = self.query_weights(points_cnl) # (1, N_points, 24)
        transforms_fwd = torch.matmul(pts_weights, dst_gtfms.view(batch_size, -1, 16)).view(batch_size, N_points, 4, 4)
        transforms_bwd = torch.inverse(transforms_fwd)
        # transforms_bwd = FastDiff4x4MinvFunction.apply(transforms_fwd.reshape(-1, 4, 4))[0].reshape_as(transforms_fwd)
        homogen_coord = torch.ones(batch_size, N_points, 1, dtype=torch.float32, device=device)
        points_homo = torch.cat([points, homogen_coord], dim=-1).view(batch_size, N_points, 4, 1)
        points_new = torch.matmul(transforms_bwd, points_homo)[..., :3, 0]
        
        return points_new, transforms_fwd, transforms_bwd, pts_weights
    
    def backward_lbs_nn_einsum(self, x, w, tfs, inverse=False):
        ''' Backward skinning based on neural network predicted skinning weights 
        Args:
            x (tensor): canonical points. shape: [B, N, D]
            w (tensor): conditional input. [B, N, J]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            x (tensor): skinned points. shape: [B, N, D]
        '''
        
        x_h = F.pad(x, (0, 1), value=1.0)

        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        w_tf_inv = torch.inverse(w_tf)
        # w_tf_inv = FastDiff4x4MinvFunction.apply(w_tf.reshape(-1, 4, 4))[0].reshape_as(w_tf)

        if inverse:
            x_h = torch.einsum("bpij,bpj->bpi", w_tf_inv, x_h)
        else:
            # x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
            x_h = torch.einsum("bpij,bpj->bpi", w_tf, x_h)

        return x_h[:, :, :3], w_tf, w_tf_inv
    
    def backward_lbs_knn(self, points, dst_gtfms, dst_vertices, skinning_weights, **kwargs):
        """Backward skinning based on nearest neighbor SMPL skinning weights

        Args:
            points (tensor): (B, N, 3)
            dst_vertices (tensor, optional): SMPL mesh vertices (6890, 3). Defaults to None.
            skinning_weights (tensor, optional): SMPL prior skinning weights. Defaults to None.
            dst_gtfms (Tensor, optional): _description_. Defaults to None.
            trans (Tensor, optional): _description_. Defaults to None.
            sdf_init_kwargs (Tensor, optional): SMPL prior initial sdf. Defaults to None.

        Returns:
            _type_: _description_
        """
        batch_size, N_points, _ = points.shape
        device = points.device
        N_knn = 3
        
        # sample skinning weights from SMPL prior weights
        knn_ret = ops.knn_points(points, dst_vertices, K=N_knn)
        p_idx, p_dists = knn_ret.idx, knn_ret.dists
        
        w = p_dists.sum(-1, True) / p_dists
        w = w / w.sum(-1, True)
        bv, _ = torch.meshgrid([torch.arange(batch_size).to(device), torch.arange(N_points).to(device)], indexing='ij')
        pts_W = 0.
        for i in range(N_knn):
            pts_W += skinning_weights[bv, p_idx[..., i], :] * w[..., i:i+1]

        transforms_fwd = torch.matmul(pts_W, dst_gtfms.view(batch_size, -1, 16)).view(batch_size, N_points, 4, 4)
        transforms_bwd = torch.inverse(transforms_fwd)
        # transforms_bwd = FastDiff4x4MinvFunction.apply(transforms_fwd.reshape(-1, 4, 4))[0].reshape_as(transforms_fwd)

        homogen_coord = torch.ones(batch_size, N_points, 1, dtype=torch.float32, device=device)
        points_homo = torch.cat([points, homogen_coord], dim=-1).view(batch_size, N_points, 4, 1)
        points_new = torch.matmul(transforms_bwd, points_homo)[:, :, :3, 0] # (1， N_points, 3)
        
        if False:
            import trimesh
            
            transforms_fwda = torch.matmul(skinning_weights, dst_gtfms.view(1, -1, 16)).view(1, 6890, 4, 4)
            transforms_bwda = torch.inverse(transforms_fwda)

            homogen_coord = torch.ones(1, 6890, 1, dtype=torch.float32, device=dst_gtfms.device)
            points_homo = torch.cat([dst_vertices, homogen_coord], dim=-1).view(1, 6890, 4, 1)
            points_newa = torch.matmul(transforms_bwda, points_homo)[:, :, :3, 0]
            trimesh.Trimesh(points_newa[0].detach().cpu().numpy()).export("tvpose_vertices_from_posed.obj")
            
        return points_new, transforms_fwd, transforms_bwd, pts_W
        
