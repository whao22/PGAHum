import os
import glob
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import argparse
from pyhocon import ConfigFactory

from torch.utils.data import DataLoader
from libs import module_config

# Arguments
parser = argparse.ArgumentParser(
    description='Inference function on with-distribution poses (ZJU training and testing).'
)
parser.add_argument('conf', type=str, help='Path to config file.')
parser.add_argument('--base_exp_dir', type=str, default=None)
parser.add_argument('--infer_mode', type=str, default='val')
parser.add_argument('--novel_pose', type=str, default=None, help='Test specified novel pose, e.g. data/data_prepared/CoreView_392')
parser.add_argument('--novel_view', type=int, default=-1, help='Test specified novel view, e.g. 1, 2, 3')
parser.add_argument('--resolution_level', type=int, default=1, help='Test rendering resolution level. e.g. 4(256, 256), 2(512, 512)')
parser.add_argument('--gpus', type=list, default=[3], help='Test on multiple GPUs.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers to use for val/test loaders.')
parser.add_argument('--run-name', type=str, default='',
                    help='Run name for Wandb logging.')

if  __name__ == '__main__':
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.conf)
    num_workers = args.num_workers
    split_mode = args.infer_mode
    conf['dataset']['res_level'] = args.resolution_level
    conf['dataset'][f'{split_mode}_subsampling_rate'] = 200
    conf['dataset'][f'{split_mode}_start_frame'] = 0
    conf['dataset'][f'{split_mode}_end_frame'] = 500
    
    # Validation dataset for novel views
    if args.novel_view >= 0:
        conf['dataset'][f'{split_mode}_views'] = [args.novel_view]
    
    # Novel-pose synthesis on training view, we determine when novel_pose is on, the split mode should not be 'val'.
    if split_mode != 'val' and args.novel_pose is not None:
        conf['dataset']['dataset'] = "zju_mocap_odp"
        conf['dataset']['new_pose_mode'] = "zjumocap"
        conf['dataset']['new_pose_folder'] = args.novel_pose
    
    dloader = DataLoader(
        module_config.get_dataset(split_mode, conf),
        batch_size=conf.train.batch_size,
        num_workers=conf.train.num_workers,
        shuffle=False)

    # Model
    model = module_config.get_model(conf, args.base_exp_dir)

    out_dir = args.base_exp_dir
    # Create logger
    latest_wandb_path = glob.glob(os.path.join(out_dir, 'wandb', 'latest-run', 'run-*.wandb'))
    if len(latest_wandb_path) == 1:
        run_id = os.path.basename(latest_wandb_path[0]).split('.')[0][4:]
    else:
        run_id = None

    if len(args.run_name) > 0:
        run_name = args.run_name
    else:
        run_name = None

    kwargs = {'settings': wandb.Settings(start_method='fork')}
    logger = pl.loggers.WandbLogger(name=run_name,
                                    project='hf-avatar',
                                    id=run_id,
                                    save_dir=out_dir,
                                    config=conf,
                                    **kwargs)

    # Create PyTorch Lightning trainer
    checkpoint_path = os.path.join(out_dir, 'checkpoints/last.ckpt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint is found!')

    trainer = pl.Trainer(logger=logger,
                        default_root_dir=out_dir,
                        accelerator='gpu',
                        devices=args.gpus,
                        num_sanity_val_steps=0)
    
    if split_mode == 'test':
        trainer.test(model=model, dataloaders=dloader, ckpt_path=checkpoint_path, verbose=True)
    elif split_mode == 'val':
        trainer.validate(model=model, dataloaders=dloader, ckpt_path=checkpoint_path, verbose=True)