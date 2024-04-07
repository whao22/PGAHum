'''
borrowed from NeuS.
'''
import mcubes
import torch
from tqdm import tqdm
import numpy as np
import mesh2sdf
import trimesh
from libs.utils.general_utils import sample_sdf_from_grid
import pytorch3d.ops as ops
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex
)


def compute_gradient(y, x, grad_outputs=None, retain_graph=True, create_graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y, requires_grad=False, device=y.device)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, retain_graph=retain_graph, create_graph=create_graph)[0]
    return grad

def lambda_sdf(points_obs, points_cnl, sdf_network, sdf_kwargs, use_init_sdf=True, scale=1):
    """Query the sdf of points. The term 'scale' means scale of scaling of delta sdf.

    Args:
        points (Tensor): points to be queried.
        sdf_network (_type_): _description_
        sdf_kwargs (_type_): _description_
        use_init_sdf (bool, optional): _description_. Defaults to True.
        scale (int or ndarray, optional): Defauts to 1 in training time, in test it 
                                          may be set to ndarray ofor each points.

    Returns:
        sdf: sdf of points
    """
    dst_vertices = sdf_kwargs['dst_vertices']
    knn_ret = ops.knn_points(points_obs.unsqueeze(0), dst_vertices, K=3)
    p_dists = (knn_ret.dists**0.5).mean(dim=-1).squeeze(0)
    
    sdf_nn_output = sdf_network(points_cnl)
    sdf = sdf_nn_output[:, :1]
    
    if use_init_sdf:
        init_sdf = sample_sdf_from_grid(points_cnl, **sdf_kwargs)
        sdf = init_sdf + sdf * scale
    sdf[p_dists>0.1] = 1
    
    return sdf

def extract_fields(bound_min, bound_max, resolution, hfavatar, sdf_kwargs, deform_kwargs, scale, device):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).to(device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).to(device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).to(device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in tqdm(enumerate(X)):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    points_obs = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    points_cnl = hfavatar.renderer.deform_points(points_obs[None], **deform_kwargs)[0].reshape(-1, 3)
                    val = lambda_sdf(points_obs, points_cnl, hfavatar.sdf_network, sdf_kwargs, scale).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, hfavatar, sdf_kwargs, deform_kwargs, scale=1, device="cpu"):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, hfavatar, sdf_kwargs, deform_kwargs, scale, device)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def prepare_smpl_sdf(vertices, volume_size):
    PADDING = 0.05
    faces = np.load('data/body_models/misc/faces.npz')['faces']

    maxv = abs(vertices).max()
    bbmax = maxv + PADDING
    bbmin = -maxv - PADDING

    vertices_norm = (vertices - bbmin) / (bbmax - bbmin + 1e-10)
    vertices_norm = (vertices_norm - 0.5) * 2
    sdf_grid = mesh2sdf.compute(vertices_norm, faces, size=volume_size)

    smpl_sdf={
        "sdf_grid": sdf_grid,
        "bbmax": bbmax,
        "bbmin": bbmin,
    }
    return smpl_sdf

def get_intersection_mask(sdf, z_vals):
    """
    sdf: n_batch, n_pixel, n_sample
    z_vals: n_batch, n_pixel, n_sample
    """
    sign = torch.sign(sdf[..., :-1] * sdf[..., 1:])
    ind = torch.min(sign * torch.arange(sign.size(2)).flip([0]).to(sign),
                    dim=2)[1]
    sign = sign.min(dim=2)[0]
    intersection_mask = sign == -1
    return intersection_mask, ind

def remove_outliers(mesh: trimesh.Trimesh, max_cluster_num: int = 1):
    """remove outlier points from mesh

    Args:
        mesh (trimesh.Trimesh): _description_
        max_cluster_num (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    mesh = mesh.as_open3d

    # cluster connected triangles
    triangles_idx_in_clusters, length_of_clusters, superficial_area_clusters = mesh.cluster_connected_triangles()
    # find the first max_cluster_num largest cluster
    max_cluster_list = sorted(length_of_clusters, reverse=True)[:max_cluster_num]
    remain_triangles = [i for i, cluster_idx in enumerate(triangles_idx_in_clusters) if length_of_clusters[cluster_idx] in max_cluster_list]
    # find all vertices in the largest cluster.
    remain_vectices = []
    for tri_idx in remain_triangles:
        tri = mesh.triangles[tri_idx]
        remain_vectices.append(tri[0])
        remain_vectices.append(tri[1])
        remain_vectices.append(tri[2])
    remain_vectices = list(set(remain_vectices))

    # remove outlier points
    mesh = mesh.select_by_index(remain_vectices)
    # # visualize the result
    # o3d.visualization.draw_geometries([mesh])
    mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)

    return mesh

def render_normal_from_mesh(verts, 
                            faces, 
                            cam_rot, 
                            cam_trans, 
                            K, 
                            mask, 
                            device, 
                            H, 
                            W, 
                            use_mask=False, 
                            cnl_render=False):
    """Render normal image from mesh.

    Args:
        verts (_type_): [B, N, 3], Vertices of the mesh.
        faces (_type_): [B, M, 3], Triangles of the mesh.
        cam_rot (_type_): [B, 3, 3], Rotation matrix of the camera.
        cam_trans (_type_): [B, 3], Translation of the camera. 
        K (_type_): [B, 3, 3], Intrinsic matrix of the camera.
        mask (_type_): [H, W], Mask of the image.
        device (_type_): _description_
        H (_type_): _description_
        W (_type_): _description_
        use_mask (bool, optional): _description_. Defaults to False.
        cnl_render (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    mesh_bar = Meshes(verts, faces)
    
    raster_settings = RasterizationSettings(
        image_size=(H, W),
    )
    image_size = torch.tensor([[H, W]], dtype=torch.float32, device=device)
    cameras = cameras_from_opencv_projection(cam_rot, cam_trans, K, image_size).to(device)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    rendered = rasterizer(mesh_bar)

    # Compute normal image in transformed space
    fg_mask = rendered.pix_to_face >= 0
    fg_faces = rendered.pix_to_face[fg_mask]
    faces_normals = -mesh_bar.faces_normals_packed().squeeze(0)
    normal_image = -torch.ones(1, H, W, 3, dtype=torch.float32, device=device)
    normal_image.masked_scatter_(fg_mask, torch.einsum('bij,pj->pi', cam_rot, faces_normals[fg_faces, :]))

    normal_image = ((normal_image + 1) / 2.0).clip(0.0, 1.0)
    normal_image = normal_image[..., [2,1,0]]
    if not use_mask:
        normal_image = normal_image[0]*255
    else:
        normal_image = normal_image[0]*255 * mask[..., None]
    alpha_map = torch.ones_like(normal_image)[..., :1] * 255
    alpha_map[normal_image[...,0]==0]=0
    normal_image = torch.concat([normal_image, alpha_map], axis=-1)
    
    return normal_image