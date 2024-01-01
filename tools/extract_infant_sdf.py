import sys
sys.path.append(".")

import os
import pickle
import numpy as np
from tools.prepare_zju_mocap.prepare_dataset import prepare_smpl_sdf
from tools.smpl.smpl_numpy import SMPL
from libs.utils.general_utils import get_02v_bone_transforms

def main(sex, model_dir, num_betas, num_poses):   
    # load skinning weights
    with open('data/body_models/smpl/male/model.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    skinning_weights = data['weights']
    
    # infant model
    poses = np.zeros(num_poses)
    betas = np.zeros(num_betas)
    infant_model = SMPL(sex=sex, model_dir=model_dir, num_betas=num_betas)
    tpose_vertices_infant, tpose_joints_infant = infant_model(poses, betas)
    
    # deform to star pose
    gtfs_02v = get_02v_bone_transforms(tpose_joints_infant)
    T = np.matmul(skinning_weights, gtfs_02v.reshape([-1, 16])).reshape([-1, 4, 4])
    vpose_vertices_infant = np.matmul(T[:, :3, :3], tpose_vertices_infant[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]
    vpose_vertices_infant = vpose_vertices_infant - vpose_vertices_infant.mean(0) # set the geometry center as origin 
    
    # comput sdf
    smpl_sdf = prepare_smpl_sdf(vertices=vpose_vertices_infant, volume_size=256)
    
    np.save(os.path.join(model_dir, sex, 'infant_sdf.npy'), 
            {'smpl_sdf': smpl_sdf,
             'vertices': vpose_vertices_infant})
    
if __name__=="__main__":
    ''' python tools/extract_infant_sdf.py '''
    sex = 'infant'
    model_dir = 'data/body_models/smpl' 
    num_poses = 72
    num_betas = 20
    
    main(sex, model_dir, num_betas, num_poses)
    print("Completed.")