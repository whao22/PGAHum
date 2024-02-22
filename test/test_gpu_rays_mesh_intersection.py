import numpy as np
import sys
sys.path.append(".")
import trimesh

from tools.gpu_rays_mesh_intersection.pycuda_ray_surface_intersect import PyCudaRSI
from tools.gpu_rays_mesh_intersection.diagnostic_input import synthesize_data
from tools.gpu_rays_mesh_intersection.diagnostic_graphics import visualise_example1



def run_example1(mode, makeplot):
    # rays
    rays = np.load("rays.npz")
    cam_loc = rays['cam_loc']
    sampled_rays = rays['sampled_rays']
    sampled_bounds_intersections = rays['sampled_bounds_intersections']
    near, far = sampled_bounds_intersections[:, 0],sampled_bounds_intersections[:, 1]
    raysFrom = sampled_rays * near + cam_loc
    raysTo = sampled_rays * far + cam_loc
    
    # mesh
    mesh = trimesh.load_mesh("posed_mesh_gt.obj")
    vertices = mesh.vertices
    triangles = mesh.faces
    
    # For "intercept_count", add a canopy (two triangular patches)
    # above the rectangular base surface to make it more interesting.
    if mode == 'intercept_count':
        vertices = np.r_[vertices, vertices[1:] + [0,0,0.2]]
        triangles = np.r_[triangles, [[5,6,8], [6,7,8]]]
        gt_crossings_detected = [0,1,2,0,2,0,0,1]
    else:
        gt_crossings_detected = [0,1,1,0,1,0,0,1]
    gt_rays = np.where(gt_crossings_detected)[0]
    gt_intercepts = np.array([[12.7,2.2,1.14], [12.9,2.4,1.21],
                                [12.6,2.9,1.23], [12.2,2.4,1.08]])
    gt_triangles = [0,1,2,3]
    gt_distances = np.sqrt(np.sum((gt_intercepts - raysFrom[gt_rays])**2, axis=1))

    if makeplot:
        visualise_example1(vertices, triangles, raysFrom, raysTo,
                            gt_rays, gt_triangles, gt_intercepts)

    #----------------------------------------------------------------------
    # Usage pattern
    params = {'USE_EXTRA_BVH_FIELDS': True, 'EPSILON': 0.001}
    cfg = {'mode': mode,
            'examine_bvh': mode=='intercept_count',
            'bvh_visualisation': [] }
    # 'examine_bvh' prints out the tree and node attributes when True.
    # 'bvh_visualisation' (an action list) must contain the word
    #    'graph' to generate a graph of the mesh triangles binary radix tree
    #    'spatial' to produce a hierarchical representation of this tree
    '''
    Note:
    - 'USE_EXTRA_BVH_FIELDS' is set to False by default.
        Both 'examine_bvh' and 'bvh_visualisation' are disabled,
        defaulting to False and [] outside of testing.
    - 'EPSILON' represents the zero threshold used in the Moller-Trumbore
        algorithm. It is set to 0.00001 by default.
    -  These options are set explicitly here for illustrative purpose.
    '''
    with PyCudaRSI(params) as pycu:
        if mode != 'barycentric':
            # Determine whether an intersection had occured
            ray_intersects = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)
        else:
            # Find intersecting [rays, distances, triangles and points]
            (intersecting_rays, distances, hit_triangles, hit_points) \
                = pycu.test(vertices, triangles, raysFrom, raysTo, cfg)
    #----------------------------------------------------------------------

    # Check the results
    if mode in ['boolean', 'intercept_count']:
        print(f'- ray_surface_intersect test {mode} results')
        for i,result in enumerate(ray_intersects):
            print(f'{i}: {result}')
        assert(np.all(ray_intersects == gt_crossings_detected))
    else:
        print(f'- ray_surface_intersect test {mode} results')
        for i in range(len(intersecting_rays)):
            p, d = hit_points[i], distances[i]
            print('ray={}: triangle={}, dist={}, point={}'.format(
                    intersecting_rays[i], hit_triangles[i], '%.6g' % d,
                    '[%.6g,%.6g,%.6g]' % (p[0],p[1],p[2])))
        #- verification
        assert(np.all(intersecting_rays == gt_rays))
        assert(np.all(hit_triangles == gt_triangles))
        assert(np.all(np.isclose(distances, gt_distances)))
        assert(np.all(np.isclose(hit_points, gt_intercepts, atol=1e-3)))
        

if __name__=="__main__":
    run_example1(mode='barycentric')