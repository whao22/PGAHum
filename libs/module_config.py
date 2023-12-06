# import libs.datasets as data
from libs.datasets2.zjumocap import ZJUMoCapDataset
from libs.hfavatar import HFAvatar
from libs.models.siren_modules import HyperBVPNet
from libs.models.deform import Deformer
import torch
from collections import OrderedDict

from libs.embeders.hannw_fourier import get_embedder
from libs.models.pose_refine import BodyPoseRefiner
from libs.models.fields_high import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from libs.models.nonrigid import NonRigidMotionMLP

from libs.models.offset import Offset
from libs.utils.network_utils import MotionBasisComputer

# Model
def get_sdf_decoder(cfg, init_weights=True):
    ''' Returns a SDF decoder instance.

    Args:
        cfg (yaml config): yaml config object
        init_weights (bool): whether to initialize the weights for the SDF network with pre-trained model (MetaAvatar)
    '''
    decoder = HyperBVPNet(**cfg['model']['decoder_kwargs'])#.to(device)
    optim_geometry_net_path = cfg['model']["decoder_kwargs"]['geometry_net']
    
    if init_weights and optim_geometry_net_path is not None:
        ckpt = torch.load(optim_geometry_net_path, map_location='cpu')

        decoder_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('decoder'):
                decoder_state_dict[k[8:]] = v

        decoder.load_state_dict(decoder_state_dict, strict=False)

    return decoder

def get_skinning_model(conf, init_weight=True):
    skinning_model = Deformer(**conf['model.skinning_model'])
    
    # init skinning model
    optim_skinning_net_path = conf.get("model.skinning_model.optim_skinning_net_path", None)
    if init_weight and optim_skinning_net_path is not None:
        ckpt = torch.load(optim_skinning_net_path, map_location='cpu')

        skinning_decoder_fwd_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('skinning_decoder_fwd'):
                skinning_decoder_fwd_state_dict[k[21:]] = v

        skinning_model.load_state_dict(skinning_decoder_fwd_state_dict, strict=False)
    return skinning_model

def get_model(conf, base_exp_dir, init_weight=True):
    total_bones = conf['general.total_bones']
    N_frames = conf['dataset.N_frames']
    
    skinning_model = get_skinning_model(conf, init_weight=init_weight)
    sdf_decoder = get_sdf_decoder(conf, init_weights=init_weight)
    pose_decoder = BodyPoseRefiner(total_bones=total_bones, **conf['model.pose_refiner'])
    motion_basis_computer = MotionBasisComputer(total_bones)
    offset_net = Offset(**conf['model.offset_net'])
    _, non_rigid_pos_embed_size = get_embedder(multires=conf.model.non_rigid.multires, 
                                               iter_val=conf.model.non_rigid.i_embed,
                                               kick_in_iter=conf.model.non_rigid.kick_in_iter,
                                               full_band_iter=conf.model.non_rigid.full_band_iter)
    non_rigid_mlp = NonRigidMotionMLP(**conf['model.non_rigid'], pos_embed_size=non_rigid_pos_embed_size)
    nerf_outside = NeRF(**conf['model.nerf'])
    deviation_network = SingleVarianceNetwork(**conf['model.variance_network'])
    color_network = RenderingNetwork(**conf['model.rendering_network'])
    sdf_network = SDFNetwork(**conf['model.sdf_network'])
    
    net_path = conf['model.nets_path']
    if init_weight and net_path is not None:
        ckpt = torch.load(net_path, map_location='cpu')
        
        # pose_decoder
        pose_decoder_fwd_state_dict = OrderedDict()
        for k, v in ckpt['network'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('pose_decoder'):
                pose_decoder_fwd_state_dict[k[13:]] = v

        pose_decoder.load_state_dict(pose_decoder_fwd_state_dict, strict=False)
        
        # non_rigid_mlp
        non_rigid_mlp_fwd_state_dict = OrderedDict()
        for k, v in ckpt['network'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('non_rigid_mlp'):
                non_rigid_mlp_fwd_state_dict[k[14:]] = v
        non_rigid_mlp.load_state_dict(non_rigid_mlp_fwd_state_dict, strict=False)
    
    return HFAvatar(conf=conf, 
                    base_exp_dir=base_exp_dir, 
                    pose_decoder=pose_decoder,
                    motion_basis_computer=motion_basis_computer,
                    offset_net=offset_net,
                    non_rigid_mlp=non_rigid_mlp,
                    nerf_outside=nerf_outside,
                    deviation_network=deviation_network,
                    color_network=color_network,
                    sdf_network=sdf_network,
                    sdf_decoder=sdf_decoder, 
                    skinning_model=skinning_model,
                    N_frames=N_frames,
                    )

# Datasets
def get_dataset(mode, cfg, view_split=None, subsampling_rate=None, start_frame=None, end_frame=None):
    ''' Returns the dataset.

    Args:
        mode (str): which mode the dataset is. Can be either train, val or test
        cfg (dict): config dictionary
        view_split (list of str): which view(s) to use. If None, will load all views
        subsampling_rate (int): frame subsampling rate for the dataset
        start_frame (int): starting frame
        end_frame (int): ending frame
    '''
    dataset_type = cfg.dataset.dataset
    dataset_folder = cfg.dataset.data_dir
    resize_img_scale = cfg.dataset.resize_img_scale
    use_aug = cfg.dataset.use_aug
    normalized_scale = cfg.dataset.normalized_scale
    backgroung_color= cfg.dataset.backgroung_color
    patch_size = cfg.dataset.patch_size
    N_patches = cfg.dataset.N_patches
    sample_subject_ratio = cfg.dataset.sample_subject_ratio
    ray_shoot_mode = cfg.dataset.ray_shoot_mode
    box_margin = cfg.dataset.box_margin
    N_samples = cfg.model.neus_renderer.n_samples
    inner_sampling = cfg.dataset.inner_sampling

    splits = {
        'train': cfg.dataset.train_split,
        'val': cfg.dataset.val_split,
        'test': cfg.dataset.test_split,
    }
    split = splits[mode]

    if view_split is None:
        view_splits = {
            'train': [str(v) for v in cfg.dataset.train_views],
            'val': [str(v) for v in cfg.dataset.val_views],
            'test': [str(v) for v in cfg.dataset.test_views],
        }
        view_split = view_splits[mode]

    if subsampling_rate is None:
        subsampling_rates = {
            'train': cfg.dataset.train_subsampling_rate,
            'val': cfg.dataset.val_subsampling_rate,
            'test': cfg.dataset.test_subsampling_rate,
        }
        subsampling_rate = subsampling_rates[mode]

    if start_frame is None:
        start_frames = {
            'train': cfg.dataset.train_start_frame,
            'val': cfg.dataset.val_start_frame,
            'test': cfg.dataset.test_start_frame,
        }
        start_frame = start_frames[mode]

    if end_frame is None:
        end_frames = {
            'train': cfg.dataset.train_end_frame,
            'val': cfg.dataset.val_end_frame,
            'test': cfg.dataset.test_end_frame,
        }
        end_frame = end_frames[mode]
    
    if dataset_type == 'zju_mocap':
        dataset = ZJUMoCapDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            patch_size=patch_size,
            N_patches=N_patches,
            sample_subject_ratio=sample_subject_ratio,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
        )
    return dataset

# Datasets
def get_dataset2(mode, cfg, view_split=None, subsampling_rate=None, start_frame=None, end_frame=None):
    ''' Returns the dataset.

    Args:
        mode (str): which mode the dataset is. Can be either train, val or test
        cfg (dict): config dictionary
        view_split (list of str): which view(s) to use. If None, will load all views
        subsampling_rate (int): frame subsampling rate for the dataset
        start_frame (int): starting frame
        end_frame (int): ending frame
    '''
    dataset_type = cfg.dataset.dataset
    dataset_folder = cfg.dataset.data_dir
    use_aug = cfg.dataset.use_aug
    normalized_scale = cfg.dataset.normalized_scale

    splits = {
        'train': cfg.dataset.train_split,
        'val': cfg.dataset.val_split,
        'test': cfg.dataset.test_split,
    }

    split = splits[mode]

    if view_split is None:
        view_splits = {
            'train': [str(v) for v in cfg.dataset.train_views],
            'val': [str(v) for v in cfg.dataset.val_views],
            'test': [str(v) for v in cfg.dataset.test_views],
        }

        view_split = view_splits[mode]

    if subsampling_rate is None:
        subsampling_rates = {
            'train': cfg.dataset.train_subsampling_rate,
            'val': cfg.dataset.val_subsampling_rate,
            'test': cfg.dataset.test_subsampling_rate,
        }

        subsampling_rate = subsampling_rates[mode]

    if start_frame is None:
        start_frames = {
            'train': cfg.dataset.train_start_frame,
            'val': cfg.dataset.val_start_frame,
            'test': cfg.dataset.test_start_frame,
        }

        start_frame = start_frames[mode]

    if end_frame is None:
        end_frames = {
            'train': cfg.dataset.train_end_frame,
            'val': cfg.dataset.val_end_frame,
            'test': cfg.dataset.test_end_frame,
        }

        end_frame = end_frames[mode]

    # Create dataset
    if dataset_type == 'people_snapshot':
        num_fg_samples = cfg.dataset.num_fg_samples
        num_bg_samples = cfg.dataset.num_bg_samples

        off_surface_thr = cfg.dataset.off_surface_thr
        inside_thr = cfg.dataset.inside_thr
        box_margin = cfg.dataset.box_margin
        sampling = cfg.dataset.sampling
        erode_mask = cfg.dataset.erode_mask
        sample_reg_surface = cfg.dataset.sample_reg_surface

        inside_weight = cfg.train.inside_weight

        high_res = cfg.dataset.high_res

        dataset = data.PeopleSnapshotDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            img_size=(540, 540) if not high_res or mode in ['val', 'test'] else (1080, 1080),
            num_fg_samples=num_fg_samples,
            num_bg_samples=num_bg_samples,
            sampling_rate=subsampling_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            off_surface_thr=off_surface_thr,
            inside_thr=inside_thr,
            box_margin=box_margin,
            sampling=sampling,
            sample_reg_surface=sample_reg_surface,
            sample_inside=inside_weight > 0,
            erode_mask=erode_mask,
        )
    elif dataset_type == 'zju_mocap':
        num_fg_samples = cfg.dataset.num_fg_samples
        num_bg_samples = cfg.dataset.num_bg_samples
        box_margin = cfg.dataset.box_margin
        sampling = cfg.dataset.sampling
        erode_mask = cfg.dataset.erode_mask
        high_res = cfg.dataset.high_res

        dataset = data.ZJUMOCAPDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            img_size=(128, 128) if not high_res or mode in ['val', 'test'] else (1024, 1024),
            num_fg_samples=num_fg_samples,
            num_bg_samples=num_bg_samples,
            sampling_rate=subsampling_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            views=view_split,
            box_margin=box_margin,
            sampling=sampling,
            erode_mask=erode_mask,
        )
    elif dataset_type == 'h36m':
        num_fg_samples = cfg.dataset.num_fg_samples
        num_bg_samples = cfg.dataset.num_bg_samples

        off_surface_thr = cfg.dataset.off_surface_thr
        inside_thr = cfg.dataset.inside_thr
        box_margin = cfg.dataset.box_margin
        sampling = cfg.dataset.sampling
        erode_mask = cfg.dataset.erode_mask
        sample_reg_surface = cfg.dataset.sample_reg_surface

        inside_weight = cfg.train.inside_weight

        dataset = data.H36MDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            img_size=(1002, 1000),
            num_fg_samples=num_fg_samples,
            num_bg_samples=num_bg_samples,
            sampling_rate=subsampling_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            views=view_split,
            off_surface_thr=off_surface_thr,
            inside_thr=inside_thr,
            box_margin=box_margin,
            sampling=sampling,
            sample_reg_surface=sample_reg_surface,
            sample_inside=inside_weight > 0,
            erode_mask=erode_mask,
        )
    elif dataset_type == 'zju_mocap_odp':
        num_fg_samples = cfg.dataset.num_fg_samples
        num_bg_samples = cfg.dataset.num_bg_samples

        box_margin = cfg.dataset.box_margin
        pose_dir = cfg.dataset.pose_dir

        dataset = data.ZJUMOCAPODPDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            pose_dir=pose_dir,
            mode=mode,
            orig_img_size=(1024, 1024),
            img_size=(512, 512),
            num_fg_samples=num_fg_samples,
            num_bg_samples=num_bg_samples,
            sampling_rate=subsampling_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            views=view_split,
            box_margin=box_margin
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg.dataset.dataset)

    return dataset