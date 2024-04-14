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
import cv2
import torch
import numpy as np
from glob import glob
import trimesh
from scipy.spatial.transform import Rotation as R

# H, W = 940, 1285
H, W = 1024, 1024
# H, W = 1080, 1080
# H, W = 1000, 1000
def render_normal_from_mesh(verts, faces, cam_rot, cam_trans, K, mask, device, cnl_render=False):
    model_outputs = {}

    verts_torch = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)
    faces_torch = torch.tensor(faces, dtype=torch.int64, device=device).unsqueeze(0)
    mesh_bar = Meshes(verts=verts_torch, faces=faces_torch)

    # TODO: image size here should be changable
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

    model_outputs.update({'output_normal': normal_image})

    normal_image = normal_image[..., [2,1,0]]
    # normal_image = normal_image.cpu().numpy()[0]*255
    
    normal_image = normal_image.cpu().numpy()[0]*255 * mask[..., None]
    alpha_map = np.ones_like(normal_image)[..., :1] * 255
    alpha_map[normal_image[...,0]==0]=0
    normal_image = np.concatenate([normal_image, alpha_map], axis=-1)
    cv2.imwrite("normal.png", normal_image)
    
    if cnl_render:
        # Render normals in canonical space
        # Frontal normal
        R, T = look_at_view_transform(2.0, 0.0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        verts_torch = torch.tensor(verts.copy(), dtype=torch.float32, device=device).unsqueeze(0)
        faces_torch = torch.tensor(faces.copy(), dtype=torch.float32, device=device).unsqueeze(0)
        mesh = Meshes(verts_torch, faces_torch)

        raster_settings = RasterizationSettings(
            image_size=1024,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        rendered = rasterizer(mesh)

        fg_mask = rendered.pix_to_face >= 0
        fg_faces = rendered.pix_to_face[fg_mask]
        faces_normals = mesh.faces_normals_packed().squeeze(0)
        normal_image = torch.zeros(1, 1024, 1024, 3, dtype=torch.float32, device=device)
        normal_image.masked_scatter_(fg_mask, faces_normals[fg_faces, :])

        normal_image = ((normal_image + 1) / 2.0).clip(0.0, 1.0)
        model_outputs.update({'normal_cano_front': normal_image})


        # Back normal
        R, T = look_at_view_transform(2.0, 0.0, 180.0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        verts_torch = torch.tensor(verts.copy(), dtype=torch.float32, device=device).unsqueeze(0)
        faces_torch = torch.tensor(faces.copy(), dtype=torch.float32, device=device).unsqueeze(0)
        mesh = Meshes(verts_torch, faces_torch)

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        rendered = rasterizer(mesh)

        fg_mask = rendered.pix_to_face >= 0
        fg_faces = rendered.pix_to_face[fg_mask]
        faces_normals = mesh.faces_normals_packed().squeeze(0)
        normal_image = torch.zeros(1, 1024, 1024, 3, dtype=torch.float32, device=device)
        normal_image.masked_scatter_(fg_mask, faces_normals[fg_faces, :])

        normal_image = ((normal_image + 1) / 2.0).clip(0.0, 1.0)
        model_outputs.update({'normal_cano_back': normal_image})


if __name__ == '__main__':
    device = torch.device('cuda:3')
    # mesh_file = "test/render_normal/data/0030.npy"
    # mesh = np.load(mesh_file, allow_pickle=True).item()
    # verts = mesh['posed_vertex']
    # faces = mesh['triangle'].astype(np.int64)
    # cam_rot = torch.tensor([[[-0.9937,  0.0937, -0.0616],
    #      [-0.0857, -0.2805,  0.9560],
    #      [ 0.0723,  0.9553,  0.2867]]], device='cuda:3')
    # cam_trans = torch.tensor([[0.0587, 0.9934, 2.9893]], device='cuda:3')
    # K = torch.tensor([[[537.1407,   0.0000, 271.4171],
    #      [  0.0000, 537.7115, 242.4418],
    #      [  0.0000,   0.0000,   1.0000]]], device='cuda:3')
    
    nuy_file = sorted(glob("exp/CoreView_394_1710683923_slurm_mvs_1_1_3_true/meshes/*.npy"))[-1]
    mesh_data = np.load(nuy_file, allow_pickle=True).item()
    camera_e = mesh_data['camera_e']
    rh = mesh_data['rh']
    th = mesh_data['th'] #+ np.array([0, -0.4, 0])
    k = mesh_data['K']
    K = torch.tensor(k, device=device).reshape(1,3,3)
    # K[:2,:2] *= 2
    extrinsic = torch.tensor(camera_e, device=device).reshape(1,4,4)
    cam_rot = extrinsic[..., :3,:3]
    # cam_rot = torch.tensor([[0,1,0],[1,0,0],[0,0,1]]).reshape_as(cam_rot).to(device)
    cam_trans = extrinsic[..., :3,3] * torch.tensor([1, 1, 1], device=device).reshape(1,3)
    
    # mesh = trimesh.load("test/eval_chd/data/377_pet-neus_gt.ply")
    # verts = mesh.vertices
    # faces = mesh.faces.astype(np.int64)
    verts = mesh_data['vertices'].astype(np.float32)
    faces = mesh_data['triangles'].astype(np.int64)
    faces = faces[..., [0,2,1]]
    # verts = np.matmul(R.from_euler('xyz', [-90, 209, 0], degrees=True).as_matrix(), verts.reshape(-1, 3, 1))[..., 0]
    verts = np.matmul(rh.T, verts.reshape(-1, 3, 1))[..., 0]
    verts = verts + th.reshape(1, 3)
    # mask = mesh_data['hit_mask'].reshape(H, W)
    mask = mesh_data['alpha_mask'].reshape(H, W)
    
    render_normal_from_mesh(verts.astype(np.float32), faces, cam_rot.float(), cam_trans, K, mask, device)