import sys
sys.path.append(".")
import pickle
from tools.smpl.smpl_numpy import SMPL
import numpy as np
import trimesh
# with open('data/body_models/smpl/male/model.pkl', 'rb') as f:
#     data = pickle.load(f, encoding='latin1')

num_betas = 20
infant_model = SMPL(sex='infant', model_dir='test', num_betas=num_betas)
smpl_model = SMPL(sex='male', model_dir='data/body_models/smpl')

poses = np.zeros(72)
betas = np.zeros(num_betas)
posed_vertices_infant, joints_infant = infant_model(poses, betas)
posed_vertices, joints = smpl_model(poses, betas[:10])

trimesh.Trimesh(posed_vertices_infant, smpl_model.faces).export("infant.obj")
trimesh.Trimesh(posed_vertices, smpl_model.faces).export("male.obj")

print()