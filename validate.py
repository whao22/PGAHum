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
    description='Validation function on with-distribution poses (ZJU training and testing).'
)
parser.add_argument('conf', type=str, help='Path to config file.')
parser.add_argument('--base_exp_dir', type=str, default=None)
parser.add_argument('--novel-pose', action='store_true', help='Test on novel-poses.')
parser.add_argument('--novel-pose-view', type=str, default=None, help='Novel view to use for rendering novel poses. Specify this argument if you only want to render a specific view of novel poses.')
parser.add_argument('--novel-view', action='store_true', help='Test on novel-views of all training poses.')
parser.add_argument('--gpus', action='store_true',default=[3], help='Test on multiple GPUs.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers to use for val/test loaders.')
parser.add_argument('--run-name', type=str, default='',
                    help='Run name for Wandb logging.')

if  __name__ == '__main__':
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.conf)
    num_workers = args.num_workers

    conf['dataset']['res_level'] = 4
    # Novel-view synthesis on training poses: evluate every 30th frame
    if args.novel_view and not args.novel_pose:
        conf['dataset']['val_subsampling_rate'] = 30

    # View-synthesis (can be either training or testing views) on novel poses
    if args.novel_pose_view is not None:
        assert (args.novel_pose)
        conf['dataset']['test_subsampling_rate'] = 1
        conf['dataset']['test_views'] = [args.novel_pose_view]

    # Dataloaders
    val_dloader = DataLoader(
        module_config.get_dataset('val', conf),
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
                        devices=[3],
                        num_sanity_val_steps=0)

    trainer.validate(model=model, dataloaders=val_dloader, ckpt_path=checkpoint_path, verbose=True)
