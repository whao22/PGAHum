import torch
import trimesh
from chamferdist import ChamferDistance
from pytorch3d.loss import chamfer_distance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pred_mesh_path = "test/test_meshes/chd.ply"
targ_mesh_path = "test/test_meshes/male_outfit2.ply"

pred_mesh = trimesh.load_mesh(pred_mesh_path)
targ_mesh = trimesh.load_mesh(targ_mesh_path)

source_cloud = torch.from_numpy(pred_mesh.vertices)[None].float().to(device) # [1, N, 3]
target_cloud = torch.from_numpy(targ_mesh.vertices)[None].float().to(device) # [1, M, 3]

# process to align the two point clouds
if False:
    source_cloud = (source_cloud - source_cloud.min()) / (source_cloud.max() - source_cloud.min() + 1e-8)
    target_cloud = (target_cloud - target_cloud.min()) / (target_cloud.max() - target_cloud.min() + 1e-8)
if 'tmp' in pred_mesh_path:
    source_cloud[..., 1] = source_cloud[..., 1] - source_cloud[..., 1].min() + 1e-8
elif 'chd' in pred_mesh_path:
    source_cloud[..., 2] = source_cloud[..., 2] - source_cloud[..., 2].min() + 1e-8
    tmp = source_cloud.clone()
    source_cloud[..., 1] = tmp[..., 2]
    source_cloud[..., 2] = tmp[..., 1]

if False:
    chamfer_dist = ChamferDistance()
    dis_foreward = chamfer_dist(source_cloud, target_cloud) / source_cloud.shape[1]
    dis_backward = chamfer_dist(target_cloud, source_cloud) / target_cloud.shape[1]
else:
    dis_foreward = chamfer_distance(source_cloud, target_cloud)
    dis_backward = chamfer_distance(target_cloud, source_cloud)

print(dis_foreward)
print(dis_backward)
