import numpy as np
import torch
import pytorch3d.ops as ops
import time
import igl

def map_points_knn(points, infant_vertices, smpl_vertices, scale, K = 10):
    # sample skinning weights from SMPL prior weights
    knn_ret_infant = ops.knn_points(points[None, ...], infant_vertices[None, ...], K=K)
    p_idx_infant = knn_ret_infant.idx.squeeze() # (M, K)
    
    knn_vertices_infant = infant_vertices[p_idx_infant] # (M, K, 3)
    knn_vertices_smpl = smpl_vertices[p_idx_infant] # (M, K, 3)
    inv_mat = torch.inverse(knn_vertices_infant @ knn_vertices_infant.permute(0, 2, 1)) # (M, K, K)
    w = points[:, None, :] @ knn_vertices_infant.permute(0, 2, 1) @ inv_mat # (M, 1, K)
    points_p = w @ knn_vertices_smpl

    return points_p

def map_points(points, infant_vertices, smpl_vertices, faces, scale, K = 10):
    device = points.device
    # detach for igl
    points = points.detach().cpu().numpy()
    infant_vertices = infant_vertices.detach().cpu().numpy()
    smpl_vertices = smpl_vertices.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    
    # closest_points - points represented the normal of the triangles
    closest_dists_infant, closest_faces_infant, closest_points_infant = \
        igl.point_mesh_squared_distance(points, infant_vertices, faces)
    v1_triangles_infant = infant_vertices[faces[closest_faces_infant, 0], :]
    v2_triangles_infant = infant_vertices[faces[closest_faces_infant, 1], :]
    v3_triangles_infant = infant_vertices[faces[closest_faces_infant, 2], :]
    
    triangles_norml_infant = np.cross(v2_triangles_infant - v1_triangles_infant, v3_triangles_infant - v1_triangles_infant)
    triangles_norml_infant = triangles_norml_infant / np.linalg.norm(triangles_norml_infant, axis=-1, keepdims=True)
    (closest_points_infant - points) / np.linalg.norm(closest_points_infant - points, axis=-1, keepdims=True)
    bary_coords = igl.barycentric_coordinates_tri(
            closest_points_infant,
            v1_triangles_infant,
            v2_triangles_infant,
            v3_triangles_infant
    )
    
    v1_triangles_smpl = smpl_vertices[faces[closest_faces_infant, 0], :]
    v2_triangles_smpl = smpl_vertices[faces[closest_faces_infant, 1], :]
    v3_triangles_smpl = smpl_vertices[faces[closest_faces_infant, 2], :]
    
    closest_points_smpl = bary_coords[:, [0]] * v1_triangles_smpl + bary_coords[:, [1]] * v2_triangles_smpl + bary_coords[:, [2]] * v3_triangles_smpl
    normal_triangles = np.cross(v2_triangles_smpl - v1_triangles_smpl, v3_triangles_smpl - v1_triangles_smpl)
    points_p = closest_points_smpl + normal_triangles * closest_dists_infant * scale
    
    return torch.Tensor(points_p).to(device)


if __name__=="__main__":
    # points = torch.rand(64*64*64, 3)
    # smpl_vertices = torch.rand(6890, 3)
    # infant_vertices = torch.rand(6890, 3)
    # faces = torch.randint(0, 6890, (13776, 3))
    # map_points(points, infant_vertices, smpl_vertices, faces, 1, K = 10)
    dicttt={}
    print(len(dicttt))