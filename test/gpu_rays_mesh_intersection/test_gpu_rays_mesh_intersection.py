import numpy as np
import sys
sys.path.append(".")
import trimesh
import os
os.chdir("libs/utils/gpu_rays_mesh_intersection")
from pycuda_ray_surface_intersect import PyCudaRSI
from diagnostic_input import synthesize_data
from diagnostic_graphics import visualise_example1
import point_cloud_utils as pcu



# def run_example1(mode, makeplot=False):
#     # rays
#     rays = np.load("../../rays.npz")
#     cam_loc = rays['cam_loc']
#     mask_erode = rays['mask_erode']
#     sampled_rays = rays['sampled_rays']
#     sampled_bounds_intersections = rays['sampled_bounds_intersections']
#     near, far = sampled_bounds_intersections[:, :1], sampled_bounds_intersections[:, 1:]
#     rays_from = sampled_rays * near + cam_loc
#     rays_to = sampled_rays * far + cam_loc
#     raysFrom = np.concatenate([rays_from, rays_to])
#     raysTo = np.concatenate([rays_to, rays_from])
    
#     # mesh
#     mesh = trimesh.load_mesh("../../posed_mesh.obj")
#     vertices = mesh.vertices
#     triangles = mesh.faces

#     #----------------------------------------------------------------------
#     # Usage pattern
#     params = {'USE_EXTRA_BVH_FIELDS': True, 'EPSILON': 0.001}
#     cfg = {'mode': mode,
#             'examine_bvh': mode=='intercept_count',
#             'bvh_visualisation': [] }
#     # 'examine_bvh' prints out the tree and node attributes when True.
#     # 'bvh_visualisation' (an action list) must contain the word
#     #    'graph' to generate a graph of the mesh triangles binary radix tree
#     #    'spatial' to produce a hierarchical representation of this tree
#     '''
#     Note:
#     - 'USE_EXTRA_BVH_FIELDS' is set to False by default.
#         Both 'examine_bvh' and 'bvh_visualisation' are disabled,
#         defaulting to False and [] outside of testing.
#     - 'EPSILON' represents the zero threshold used in the Moller-Trumbore
#         algorithm. It is set to 0.00001 by default.
#     -  These options are set explicitly here for illustrative purpose.
#     '''
#     with PyCudaRSI(params) as pycu:
#         if mode != 'barycentric':
#             # Determine whether an intersection had occured
#             ray_intersects = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)
#         else:
#             # Find intersecting [rays, distances, triangles and points]
#             (intersecting_rays, distances, hit_triangles, hit_points) \
#                 = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)
#     #----------------------------------------------------------------------

#     # # Check the results
#     # if mode in ['boolean', 'intercept_count']:
#     #     print(f'- ray_surface_intersect test {mode} results')
#     #     for i,result in enumerate(ray_intersects):
#     #         print(f'{i}: {result}')
#     #     assert(np.all(ray_intersects == gt_crossings_detected))
#     # else:
#     #     print(f'- ray_surface_intersect test {mode} results')
#     #     for i in range(len(intersecting_rays)):
#     #         p, d = hit_points[i], distances[i]
#     #         print('ray={}: triangle={}, dist={}, point={}'.format(
#     #                 intersecting_rays[i], hit_triangles[i], '%.6g' % d,
#     #                 '[%.6g,%.6g,%.6g]' % (p[0],p[1],p[2])))

#     # visulization
#     rays_end = np.concatenate([raysFrom, raysTo, np.tile(cam_loc[None, :], (2048, 1))])
#     trimesh.Trimesh(rays_end).export("points.obj")
#     trimesh.Trimesh(hit_points).export("hit_points.obj")
#     mesh.export("smpl_mesh.obj")
    
def run_example1():
    # rays
    rays = np.load("/home/wanghao/workspace/hf-avatar/rays.npz")
    cam_loc = rays['cam_loc']
    mask_erode = rays['mask_erode']
    sampled_rays = rays['sampled_rays']
    sampled_bounds_intersections = rays['sampled_bounds_intersections']
    near, far = sampled_bounds_intersections[:, :1], sampled_bounds_intersections[:, 1:]
    near_bbox_points = sampled_rays * near + cam_loc
    far_bbox_points = sampled_rays * far + cam_loc
    rays_o = np.concatenate([near_bbox_points, far_bbox_points])
    rays_d = np.concatenate([sampled_rays, -sampled_rays])
        
    # mesh
    mesh = trimesh.load_mesh("/home/wanghao/workspace/hf-avatar/posed_mesh.obj")
    vertices = mesh.vertices
    triangles = mesh.faces

    fid, bc, t = pcu.ray_mesh_intersection(vertices.astype(np.float32), triangles.astype(int), rays_o, rays_d)
    hit_points = pcu.interpolate_barycentric_coords(triangles, fid, bc, vertices)
    hit_mask = fid > 0 
    hit_triangles = fid
    
    mask_f = hit_mask[:len(sampled_rays)]
    mask_b = hit_mask[len(sampled_rays):]
    mask_hit = mask_f * mask_b
    hit_points_f = hit_points[:len(sampled_rays)][mask_hit]
    hit_points_b = hit_points[len(sampled_rays):][mask_hit]
    hit_points = np.stack([hit_points_f, hit_points_b], axis=0)

    hit_triangles_f = triangles[hit_triangles[:len(sampled_rays)][mask_hit]]
    hit_triangles_b = triangles[hit_triangles[len(sampled_rays):][mask_hit]]
    hit_triangles = np.stack([hit_triangles_f, hit_triangles_b], axis=0)
    # visulization
    # rays_end = np.concatenate([raysFrom, raysTo, np.tile(cam_loc[None, :], (2048, 1))])
    # trimesh.Trimesh(rays_end).export("points.obj")
    trimesh.Trimesh(hit_points).export("hit_points.obj")
    mesh.export("smpl_mesh.obj")
    
    
if __name__=="__main__":
    run_example1()