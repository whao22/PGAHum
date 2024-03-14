import os
import torch
import lpips
import cv2
import logging
import numpy as np
from typing import Optional
from collections import OrderedDict
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from libs.utils.metrics import pytorch_ssim
from libs.utils.general_utils import augm_rots, feasible
from libs.utils.network_utils import set_requires_grad
from libs.utils.metrics.metrics import psnr_metric, ssim_metric, lpips_metric
from libs.renderers.renderer import IDHRenderer
from libs.renderers.loss import IDHRLoss

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

def mse2psnr(mse):
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)

class HFAvatar(pl.LightningModule):
    def __init__(self, 
                conf,
                models: dict):
        super().__init__()
        
        self.conf = conf
        self.lr = conf.train.lr
        self.base_exp_dir = conf.general.base_exp_dir
        self.warm_up_end = conf.train.warm_up_end
        self.anneal_end = conf.train.anneal_end
        self.total_bones = conf.general.total_bones
        self.batch_size = conf.train.batch_size
        self.N_rays = conf.dataset.patch_size**2 * conf.dataset.N_patches
        self.resolution_level = conf.dataset.res_level
        
        self.sdf_mode = conf.model.sdf_mode
        self.deform_mode = conf.model.deform_mode
        self.inner_sampling = conf.dataset.inner_sampling
        
        logging.info(f"Base experiment directory is {self.base_exp_dir}")
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        # Loss Function
        self.lpips = lpips.LPIPS(net='vgg')
        set_requires_grad(self.lpips, requires_grad=False)
        self.criteria = IDHRLoss(**conf['train.weights'],
                                 rgb_loss_type=conf.train.rgb_loss_type,
                                 lpips = self.lpips,
                                 weight_decay_end = conf.train.weight_decay_end,
                                 kick_out_iter_skinning = conf.model.skinning_model.kick_out_iter)
        
        # Networks
        nets_to_train = {}
        self.pose_decoder = models['pose_decoder']
        self.motion_basis_computer = models['motion_basis_computer']
        self.deviation_network = models['deviation_network']
        self.color_network = models['color_network']
        nets_to_train.update({
            "pose_decoder": self.pose_decoder,
            "motion_basis_computer": self.motion_basis_computer,
            "deviation_network": self.deviation_network,
            "color_network": self.color_network,
        })
        
        # Deformer
        if self.deform_mode == 0:
            self.offset_net = models['offset_net']
            self.non_rigid_mlp = models['non_rigid_mlp']
            nets_to_train.update({
                "offset_net": self.offset_net,
                "non_rigid_mlp": self.non_rigid_mlp,
            })
        elif self.deform_mode == 1:
            self.skinning_model = models['skinning_model']
            nets_to_train.update({
                "skinning_model": self.skinning_model,
            })
        
        # NeRF for outside rendering
        if self.conf.model.neus_renderer.n_outside > 0:
            self.nerf_outside = models['nerf_outside']
            nets_to_train.update({
                "nerf_outside": self.nerf_outside,
            })
        
        # SDF network
        if self.sdf_mode == 0 or self.sdf_mode == 1:
            self.sdf_network = models['sdf_network']
            nets_to_train.update({
                "sdf_network": self.sdf_network,
            })
        elif self.sdf_mode == 2:
            self.latent = nn.Embedding(conf.dataset.N_frames, 128)
            self.sdf_decoder = models['sdf_decoder']
            nets_to_train.update({
                'sdf_decoder_net': self.sdf_decoder.net.layers,
                'sdf_decoder_pose_encoder': self.sdf_decoder.pose_encoder
            })
        
        self.nets_to_train = nets_to_train
        
        # Renderer
        self.renderer = IDHRenderer(conf = self.conf,
                                    models = self.nets_to_train,
                                    total_bones = self.total_bones,
                                    sdf_mode = self.sdf_mode,
                                    deform_mode = self.deform_mode,
                                    inner_sampling = self.inner_sampling,
                                    use_init_sdf = conf.model.use_init_sdf,
                                    sdf_threshold = conf.model.sdf_threshold,
                                    N_iter_backward = conf.model.N_iter_backward,
                                    **self.conf.model.neus_renderer)
    
    def get_sdf_decoder(self, inputs, idx, eval=False):
        view_input_noise = self.conf.train.view_input_noise
        pose_input_noise = self.conf.train.pose_input_noise
        nv_noise_type = self.conf.train.nv_noise_type
        
        rots = inputs['dst_Rs']
        Jtrs = inputs['tjoints']
        batch_size = rots.size(0)
        device = rots.device

        decoder_input = {'coords': torch.zeros(1, 1, 3, dtype=torch.float32, device=device), 'rots': rots[0].reshape(-1, 9).unsqueeze(0), 'Jtrs': Jtrs[0].unsqueeze(0)}
        if 'geo_latent_code_idx' in inputs.keys():
            geo_latent_code = self.latent(inputs['geo_latent_code_idx'])
            decoder_input.update({'latent': geo_latent_code})

        # Do augmentation to input poses and views, if applicable
        if (pose_input_noise or view_input_noise) and not eval:
            if np.random.uniform() <= 0.5:
                if pose_input_noise:
                    decoder_input['rots_noise'] = torch.normal(mean=0, std=0.1, size=rots.shape, dtype=rots.dtype, device=device)
                    inputs['pose_cond']['rot_noise'] = torch.normal(mean=0, std=0.1, size=(batch_size,9), dtype=rots.dtype, device=device)
                    inputs['pose_cond']['trans_noise'] = torch.normal(mean=0, std=0.1, size=(batch_size,3), dtype=rots.dtype, device=device)

                if view_input_noise:
                    if nv_noise_type == 'gaussian':
                        inputs['pose_cond']['view_noise'] = torch.normal(mean=0, std=0.1, size=inputs['ray_dirs'].shape, dtype=rots.dtype, device=device)
                    elif nv_noise_type == 'rotation':
                        inputs['pose_cond']['view_noise'] = torch.tensor(augm_rots(45, 45, 45), dtype=torch.float32, device=device).unsqueeze(0)
                    else:
                        raise ValueError('wrong nv_noise_type, expected either gaussian or rotation, got {}'.format(self.nv_noise_type))
        
        # Generate SDF network from hypernetwork (MetaAvatar)
        output = self.sdf_decoder(decoder_input) 
        sdf_decoder = output['decoder']
        sdf_params = output['params']
        return sdf_decoder, sdf_params
    
    def training_step(self, batch, idx):
        if self.sdf_mode == 2:
            sdf_decoder, sdf_params = self.get_sdf_decoder(batch, idx)
        
        # forward
        render_out = self.renderer.render(batch, 
                                          self.global_step, 
                                          cos_anneal_ratio = self.get_cos_anneal_ratio(), 
                                          sdf_decoder = sdf_decoder if self.sdf_mode == 2 else None)
        loss_results = self.criteria(render_out,
                                     batch,
                                     self.global_step,
                                     sdf_params if self.sdf_mode == 2 else None)
        
        # loss
        self.log('loss_color', loss_results['loss_color'])
        self.log('loss_mask', loss_results['loss_mask'])
        self.log('loss_eikonal', loss_results['loss_eikonal'])
        self.log('loss_pips', loss_results['loss_pips'])
        self.log('loss_mse', loss_results['loss_mse'])
        self.log('loss_nssim', loss_results['loss_nssim'])
        self.log('loss_skinning_weights', loss_results['loss_skinning_weights'])
        self.log('loss_params', loss_results['loss_params'])
        self.log('loss_pose_refine', loss_results['loss_pose_refine'])
        
        self.log('loss', loss_results['loss'])
        
        return loss_results['loss']
    
    def test_step(self, data, idx):
        if self.sdf_mode == 2:
            sdf_decoder, sdf_params = self.get_sdf_decoder(data, idx)
        
        with torch.enable_grad():
            out_rgb_fine = []
            out_normal_fine = []
            
            batch_rays_list = torch.split(data['batch_rays'], self.N_rays, dim=1)
            z_vals_list = torch.split(data['z_vals'], self.N_rays, dim=1)
            hit_mask_list = torch.split(data['hit_mask'], self.N_rays, dim=1)
            
            for batch_rays, z_vals, hit_mask in zip(batch_rays_list, z_vals_list, hit_mask_list):
                data_batch = data.copy()
                data_batch['batch_rays'] = batch_rays
                data_batch['z_vals'] = z_vals
                data_batch['hit_mask'] = hit_mask
                
                render_out = self.renderer.render(data_batch, 
                                                    self.global_step, 
                                                    cos_anneal_ratio = self.get_cos_anneal_ratio(), 
                                                    sdf_decoder = sdf_decoder if self.sdf_mode == 2 else None)

                if feasible('color', render_out):
                    out_rgb_fine.append(render_out['color'].detach())
                if feasible('gradients', render_out) and feasible('weights', render_out):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere', render_out):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).detach()
                    out_normal_fine.append(normals)
                    
                torch.cuda.empty_cache()
                del render_out
                del data_batch
            
        with torch.no_grad():
            H, W, E = data['height'].squeeze(), data['width'].squeeze(), data['extrinsic'].squeeze()
            bg_color = data['background_color'].squeeze()
            inter_mask = data['hit_mask'].squeeze().reshape(H, W)
            
            image_pred = torch.tile(bg_color[None, None, :], (H, W, 1))
            if len(out_rgb_fine) > 0:
                img_fine = torch.cat(out_rgb_fine, dim=0)
                image_pred[inter_mask] = img_fine[inter_mask.reshape(-1)]
                image_pred = (image_pred * 256).clip(0, 255)

            normal_image = torch.tile(bg_color[None, None, :], (H, W, 1))
            if len(out_normal_fine) > 0:
                normal_img = torch.cat(out_normal_fine, dim=0)
                normal_image[inter_mask] = normal_img[inter_mask.reshape(-1)]
                rot = torch.inverse(E[:3,:3])
                normal_image = (torch.matmul(rot[None, ...], normal_image[..., None])[..., 0] * 128 + 128).clip(0, 255)

            # log
            self.logger.log_image(key="test_samples",
                    images=[image_pred.detach().cpu().numpy(),
                            normal_image.detach().cpu().numpy()],
                    caption=["pred image", "normal map"]
            )
    
    def validation_step(self, data, idx):
        if self.sdf_mode == 'hyper_net':
            sdf_decoder, sdf_params = self.get_sdf_decoder(data, idx)
        
        with torch.enable_grad():
            out_rgb_fine = []
            out_normal_fine = []
            metrics_dict = {}
            
            batch_rays_list = torch.split(data['batch_rays'], self.N_rays, dim=1)
            z_vals_list = torch.split(data['z_vals'], self.N_rays, dim=1)
            hit_mask_list = torch.split(data['hit_mask'], self.N_rays, dim=1)
            
            for batch_rays, z_vals, hit_mask in zip(batch_rays_list, z_vals_list, hit_mask_list):
                data_batch = data.copy()
                data_batch['batch_rays'] = batch_rays
                data_batch['z_vals'] = z_vals
                data_batch['hit_mask'] = hit_mask
                
                render_out = self.renderer.render(data_batch, 
                                                    self.global_step, 
                                                    cos_anneal_ratio = self.get_cos_anneal_ratio(), 
                                                    sdf_decoder = sdf_decoder if self.sdf_mode == 2 else None)

                if feasible('color', render_out):
                    out_rgb_fine.append(render_out['color'].detach())
                if feasible('gradients', render_out) and feasible('weights', render_out):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere', render_out):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).detach()
                    out_normal_fine.append(normals)
                    
                torch.cuda.empty_cache()
                del render_out
                del data_batch
        
        with torch.no_grad():
            img_fine = None
            normal_img = None
            H, W, E = data['height'].squeeze(), data['width'].squeeze(), data['extrinsic'].squeeze()
            bg_color = data['background_color'].squeeze()
            inter_mask = data['hit_mask'].squeeze().reshape(H, W)
            
            image_targ = torch.tile(bg_color[None, None, :], (H, W, 1))
            img_true = data['batch_rays'][..., 6:9].squeeze()
            image_targ[inter_mask] = img_true[inter_mask.reshape(-1)]
            image_targ = image_targ * 255
            
            image_pred = torch.tile(bg_color[None, None, :], (H, W, 1))
            if len(out_rgb_fine) > 0:
                img_fine = torch.cat(out_rgb_fine, dim=0)
                image_pred[inter_mask] = img_fine[inter_mask.reshape(-1)]
                image_pred = (image_pred * 256).clip(0, 255)
                
                bbox_mask = data['bbox_mask'].squeeze().detach().cpu().numpy() # (H, W), ndarray
                x, y, w, h = cv2.boundingRect(bbox_mask.astype(np.uint8))
                image_pred_ = image_pred[y:y+h, x:x+w] / 255
                image_targ_ = image_targ[y:y+h, x:x+w] / 255
                
                psnr = mse2psnr(F.mse_loss(image_pred_, image_targ_).item()) # 27；31
                ssim = pytorch_ssim.ssim(image_pred_.permute(2, 0, 1).unsqueeze(0), image_targ_.permute(2, 0, 1).unsqueeze(0)).item()
                lpips = self.lpips(image_pred_.permute(2, 0, 1).unsqueeze(0), image_targ_.permute(2, 0, 1).unsqueeze(0)).item()
                metrics_dict.update({
                    'psnr': psnr,
                    'ssim': ssim,
                    'lpips': lpips
                })
                print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}")

                self.logger.log_image(key="cropped_images",
                    images=[(image_pred_*255).detach().cpu().numpy(), 
                            (image_targ_*255).detach().cpu().numpy()],
                    caption=["pred image", 'GT image']
                )
            
            normal_image = torch.tile(bg_color[None, None, :], (H, W, 1))
            if len(out_normal_fine) > 0:
                normal_img = torch.cat(out_normal_fine, dim=0)
                normal_image[inter_mask] = normal_img[inter_mask.reshape(-1)]
                rot = torch.inverse(E[:3,:3])
                normal_image = (torch.matmul(rot[None, ...], normal_image[..., None])[..., 0] * 128 + 128).clip(0, 255)
            
            # log
            self.logger.log_image(key="validation_samples",
                    images=[image_pred.detach().cpu().numpy(), 
                            image_targ.detach().cpu().numpy(), 
                            normal_image.detach().cpu().numpy()],
                    caption=["pred image", 'GT image',"normal map"]
            )
            
        return metrics_dict
    
    def validation_epoch_end(self, validation_step_outputs):
        psnr, ssim, lpips = [], [], []
        for output in validation_step_outputs:
            if len(output) >= 1:
                psnr.append(output['psnr'])
                ssim.append(output['ssim'])
                lpips.append(output['lpips'])

        if len(psnr) >= 1:
            psnr = np.array(psnr).mean()
            ssim = np.array(ssim).mean()
            lpips = np.array(lpips).mean()
            
            self.log('val_psnr', psnr)
            self.log('val_ssim', ssim)
            self.log('val_lpips', lpips)
    
    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        self.update_learning_rate(optimizer)
        
    def update_learning_rate(self, optimizer):
        decay_rate = 0.1
        decay_steps = self.warm_up_end * 1000
        learning_factor = decay_rate ** (self.global_step / decay_steps)
        
        for param_group in optimizer.param_groups:
            if f"lr_{param_group['name']}" in self.conf.train:
                base_lr = self.conf.train[f'lr_{param_group["name"]}']
                new_lr = base_lr * learning_factor
            else:
                new_lr = self.lr * learning_factor
            param_group['lr'] = new_lr
        self.log('lr', new_lr)
    
    def configure_optimizers(self):
        params_to_train = []
        
        for model in self.nets_to_train:    
            lr = self.conf.train.get_float(f'lr_{model}', 0)
            params_to_train += [{
                "params": self.nets_to_train[model].parameters(),
                "lr": lr if lr!= 0.0 else self.lr,
                "name": model
            }]
        
        optimizer = torch.optim.Adam(params_to_train, lr=self.lr, betas=(0.9, 0.999))
        return optimizer
    
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.global_step / self.anneal_end])