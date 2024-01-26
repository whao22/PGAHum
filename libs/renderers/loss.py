import torch
import lpips
from torch import nn
from torch.nn import functional as F



class IDHRLoss(nn.Module):
    ''' Loss class for Implicit Differentiable Human Renderer (IDHR) '''

    def __init__(self, 
                 rgb_weight=0., 
                 perceptual_weight=0., 
                 eikonal_weight=0., 
                 mask_weight=0., 
                 skinning_weight=0.,
                 params_weight=0.,
                 pose_refine_weight=0.,
                 rgb_loss_type='l1',
                 lpips = None):
        """initialize loss class for loss computing. 

        Args:
            rgb_weight (_type_, optional): _description_. Defaults to 0..
            perceptual_weight (_type_, optional): _description_. Defaults to 0..
            eikonal_weight (_type_, optional): _description_. Defaults to 0..
            mask_weight (_type_, optional): _description_. Defaults to 0..
            skinning_weight (_type_, optional): _description_. Defaults to 0..
            rgb_loss_type (str, optional): _description_. Defaults to 'l1'.

        Raises:
            ValueError: _description_
        """

        super().__init__()
        self.rgb_weight = rgb_weight
        self.perceptual_weight = perceptual_weight
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.skinning_weight = skinning_weight
        self.params_weight = params_weight
        self.pose_refine_weight = pose_refine_weight
        
        if rgb_loss_type == 'l1':
            self.l1_loss = nn.L1Loss(reduction='mean')
        elif rgb_loss_type == 'mse':
            self.l1_loss = nn.MSELoss(reduction='mean')
        elif rgb_loss_type == 'smoothed_l1':
            self.l1_loss = nn.SmoothL1Loss(reduction='mean', beta=1e-1)
        else:
            raise ValueError('Unsupported RGB loss type: {}. Only l1, smoothed_l1 and mse are supported'.format(rgb_loss_type))
        
        if self.perceptual_weight > 0.:
            self.p_loss = lpips
    
    def scale_for_lpips(self, image_tensor):
        return image_tensor * 2. - 1.

    def get_rgb_loss(self, color_pre, color_gt):
        rgb_loss = self.l1_loss(color_pre, color_gt)
        return rgb_loss

    def unpack_image(self, predicted, target):
        bg_color = target['background_color'][0]
        div_indices = target['patch_div_indices'][0]
        patch_masks = target['patch_masks'][0]
        target_patches = target['target_patches'][0]
        N_patch = len(div_indices) - 1
        pre_color = predicted['color']
        
        assert patch_masks.shape[0] == N_patch
        assert target_patches.shape[0] == N_patch

        patch_imgs = bg_color.expand(target_patches.shape).clone() # (N_patch, H, W, 3)
        for i in range(N_patch):
            patch_imgs[i, patch_masks[i].bool()] = pre_color[div_indices[i]:div_indices[i+1]]

        return patch_imgs, target_patches
    
    def get_perceptual_loss(self, color_pre, color_gt):
        pred_patch = self.scale_for_lpips(color_pre.permute(0, 3, 1, 2))
        gt_patch = self.scale_for_lpips(color_gt.permute(0, 3, 1, 2))
        perceptual_loss = self.p_loss(pred_patch, gt_patch).mean()
        return perceptual_loss

    def get_skinning_weights_loss(self, pts_w_pre, pts_w_gt):
        return torch.abs(pts_w_pre - pts_w_gt).sum(-1).mean()

    def get_mask_loss(self, device):
        return torch.zeros(1, device=device)

    def get_sdf_params_loss(self, sdf_params):
        sdf_params = torch.cat(sdf_params, dim=1)
        n_params = sdf_params.size(-1)

        return sdf_params.norm(dim=-1).mean() / n_params

    def forward(self, model_outputs, ground_truth, sdf_params):
        if self.perceptual_weight > 0 :
            color_pre, color_gt = self.unpack_image(model_outputs, ground_truth)
        else:
            color_gt = ground_truth['batch_rays'][..., 6:9][0]
            color_pre = model_outputs['color']
            fg_mask = ground_truth['batch_rays'][..., 9:10][0]
            color_pre = color_pre * fg_mask
            color_gt = color_gt * fg_mask

        pts_w_gt = model_outputs['pts_W_sampled']
        pts_w_pre = model_outputs['pts_W_pred']
        gradient_error = model_outputs['gradient_error']
        pose_refine_error = model_outputs['pose_refine_error']
        device = color_pre.device
        
        # compute each loss item
        if self.rgb_weight > 0:
            loss_color = self.get_rgb_loss(color_pre, color_gt)
        else:
            loss_color = torch.zeros(1, device=device)
        
        if self.perceptual_weight > 0:
            loss_pips = self.get_perceptual_loss(color_pre, color_gt)
        else:
            loss_pips = torch.zeros(1, device=device)
        
        if self.skinning_weight > 0 and pts_w_pre is not None:
            loss_skinning_weights = self.get_skinning_weights_loss(pts_w_pre, pts_w_gt)
        else:
            loss_skinning_weights = torch.zeros(1, device=device)

        if self.eikonal_weight > 0:
            loss_eikonal = gradient_error
        else:
            loss_eikonal = torch.zeros(1, device=device)
        
        if self.mask_weight > 0:
            loss_mask = self.get_mask_loss(device)
        else:
            loss_mask = torch.zeros(1, device=device)
        
        if self.params_weight > 0 and sdf_params is not None:
            loss_params = self.get_sdf_params_loss(sdf_params)
        else:
            loss_params = torch.zeros(1, device=device)
        
        if self.pose_refine_weight > 0 and pose_refine_error is not None:
            loss_pose_refine = pose_refine_error
        else:
            loss_pose_refine = torch.zeros(1, device=device)
        
        # get the final loss
        loss_color = loss_color * self.rgb_weight 
        loss_pips = loss_pips * self.perceptual_weight
        loss_skinning_weights = loss_skinning_weights * self.skinning_weight
        loss_eikonal = loss_eikonal * self.eikonal_weight
        loss_mask = loss_mask * self.mask_weight
        loss_params = loss_params * self.params_weight
        loss_pose_refine = loss_pose_refine * self.pose_refine_weight
        
        loss = loss_color + loss_pips + loss_skinning_weights + loss_eikonal + loss_mask + loss_params + loss_pose_refine
        
        loss_results = {
            "loss": loss,
            "loss_color": loss_color,
            "loss_pips": loss_pips,
            "loss_skinning_weights": loss_skinning_weights,
            "loss_eikonal": loss_eikonal,
            "loss_mask": loss_mask,
            "loss_params": loss_params,
            "loss_pose_refine": loss_pose_refine,
        }
        
        return loss_results