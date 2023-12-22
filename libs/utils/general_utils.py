import torch
import cv2
import numpy as np
import point_cloud_utils as pcu
import pytorch3d
from scipy.spatial.transform import Rotation
from libs.utils.MCAcc import GridSamplerMine3dFunction
# from libs.utils.grid_sample_3d import grid_sample_3d
# from libs.utils.ops.grid_sample import grid_sample_3d as grid_sample_3d_cuda

def sample_sdf_from_grid(points, cnl_grid_sdf, cnl_bbmin, cnl_bbmax):
    ori_shape = points.shape
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    if len(cnl_grid_sdf.shape) !=5:
        cnl_grid_sdf = cnl_grid_sdf.unsqueeze(0)
    
    B, N, _ = points.shape
    points = (points - cnl_bbmin) / (cnl_bbmax - cnl_bbmin + 1e-10)
    points = (points - 0.5) * 2
    
    points = points.view(B, N, 1, 1, 3)
    points_sampled = points.clone()
    points_sampled[..., 0] = points[..., 2]
    points_sampled[..., 1] = points[..., 1]
    points_sampled[..., 2] = points[..., 0]
    # sdf_sampled = grid_sample_3d(cnl_grid_sdf, points_sampled)
    # sdf_sampled = grid_sample_3d_cuda(cnl_grid_sdf, points_sampled)
    # sdf_sampled = torch.nn.functional.grid_sample(cnl_grid_sdf, points_sampled)
    sdf_sampled = GridSamplerMine3dFunction.apply(cnl_grid_sdf, points_sampled)
    sdf_sampled = sdf_sampled / 2. * (cnl_bbmax - cnl_bbmin)
    return sdf_sampled.view(list(ori_shape)[:-1]+[1])

def sample_sdf(points, sdf_model, cnl_grid_sdf=None, out_feature=False, **kwargs):
    if cnl_grid_sdf is not None:
        init_sdf = sample_sdf_from_grid(points, cnl_grid_sdf, **kwargs)
    else:
        init_sdf = 0.
        
    if not out_feature:
        delta_sdf = sdf_model(points)
        return init_sdf + delta_sdf
    else:
        feature_vecs = sdf_model[:-1](points).squeeze(0)
        delta_sdf = sdf_model[-1](feature_vecs).squeeze(0) 
        return feature_vecs, init_sdf + delta_sdf

def augm_rots(roll_range=90, pitch_range=90, yaw_range=90):
    """ Get augmentation for rotation matrices.

    Args:
        roll_range (int): roll angle sampling range (train mode) or value (test mode)
        pitch_range (int): pitch angle sampling range (train mode) or value (test mode)
        yaw_range (int): yaw angle sampling range (train mode) or value (test mode)

    Returns:
        rot_mat (4 x 4 float numpy array): homogeneous rotation augmentation matrix.
    """
    # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
    # Roll
    rot_x = min(2*roll_range,
            max(-2*roll_range, np.random.randn()*roll_range))

    sn, cs = np.sin(np.pi / 180 * rot_x), np.cos(np.pi / 180 * rot_x)
    rot_x = np.eye(3)
    rot_x[1, 1] = cs
    rot_x[1, 2] = -sn
    rot_x[2, 1] = sn
    rot_x[2, 2] = cs

    rot_y = min(2*pitch_range,
            max(-2*pitch_range, np.random.rand()*pitch_range))

    # Pitch
    sn, cs = np.sin(np.pi / 180 * rot_y), np.cos(np.pi / 180 * rot_y)
    rot_y = np.eye(3)
    rot_y[0, 0] = cs
    rot_y[0, 2] = sn
    rot_y[2, 0] = -sn
    rot_y[2, 2] = cs

    rot_z = min(2*yaw_range,
            max(-2*yaw_range, np.random.randn()*yaw_range))

    # Yaw
    sn, cs = np.sin(np.pi / 180 * rot_z), np.cos(np.pi / 180 * rot_z)
    rot_z = np.eye(3)
    rot_z[0, 0] = cs
    rot_z[0, 1] = -sn
    rot_z[1, 0] = sn
    rot_z[1, 1] = cs

    rot_mat = np.dot(rot_x, np.dot(rot_y, rot_z))

    return rot_mat

def get_params_by_key(model, key, exclude=False):
    if exclude:
        for name, param in model.named_parameters():
            if name != key:
                yield param
    else:
        for name, param in model.named_parameters():
            if name == key:
                yield param

''' Functions copied from Neural Body
'''
def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    # near = near[mask_at_box] / norm_d[mask_at_box, 0]
    # far = far[mask_at_box] / norm_d[mask_at_box, 0]
    near = near / norm_d[..., 0]
    far = far / norm_d[..., 0]
    return near, far, mask_at_box

def normalize(x):
    return x / np.linalg.norm(x)

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts-c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def gen_path(RT, num_render_views=50, center=None):
    lower_row = np.array([[0., 0., 0., 1.]])

    # transfer RT to camera_to_world matrix
    RT = np.array(RT)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1],
                         -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    if center is None:
        center = RT[:, :3, 3].mean(0)
        z_off = 1.3

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, num_render_views + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world -
                      np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1],
                              -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return render_w2c

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    prob_all = torch.ones(n_batch * n_point, 24, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [24]]) * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))

    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all


''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    prob_all = torch.ones(n_batch * n_point, 24, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [24]]) * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))

    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all

def augm_rots(roll_range=90, pitch_range=90, yaw_range=90):
    """ Get augmentation for rotation matrices.

    Args:
        roll_range (int): roll angle sampling range (train mode) or value (test mode)
        pitch_range (int): pitch angle sampling range (train mode) or value (test mode)
        yaw_range (int): yaw angle sampling range (train mode) or value (test mode)

    Returns:
        rot_mat (4 x 4 float numpy array): homogeneous rotation augmentation matrix.
    """
    # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
    # Roll
    rot_x = min(2*roll_range,
            max(-2*roll_range, np.random.randn()*roll_range))

    sn, cs = np.sin(np.pi / 180 * rot_x), np.cos(np.pi / 180 * rot_x)
    rot_x = np.eye(3)
    rot_x[1, 1] = cs
    rot_x[1, 2] = -sn
    rot_x[2, 1] = sn
    rot_x[2, 2] = cs

    rot_y = min(2*pitch_range,
            max(-2*pitch_range, np.random.rand()*pitch_range))

    # Pitch
    sn, cs = np.sin(np.pi / 180 * rot_y), np.cos(np.pi / 180 * rot_y)
    rot_y = np.eye(3)
    rot_y[0, 0] = cs
    rot_y[0, 2] = sn
    rot_y[2, 0] = -sn
    rot_y[2, 2] = cs

    rot_z = min(2*yaw_range,
            max(-2*yaw_range, np.random.randn()*yaw_range))

    # Yaw
    sn, cs = np.sin(np.pi / 180 * rot_z), np.cos(np.pi / 180 * rot_z)
    rot_z = np.eye(3)
    rot_z[0, 0] = cs
    rot_z[0, 1] = -sn
    rot_z[1, 0] = sn
    rot_z[1, 1] = cs

    rot_mat = np.dot(rot_x, np.dot(rot_y, rot_z))

    return rot_mat

def normalize_canonical_points(pts, bbmin, bbmax):
    """normalize canonical points into specified bounding box

    Args:
        pts (_type_): _description_
        coord_min (_type_): _description_
        coord_max (_type_): _description_
        center (_type_): _description_

    Returns:
        _type_: _description_
    """
    padding = (bbmax - bbmin) * 0.05
    pts = (pts - bbmin + padding) / (bbmax - bbmin) / 1.1
    pts = pts - 0.5
    pts = pts * 2.

    return pts

def get_02v_bone_transforms(Jtr, rot45p=None, rot45n=None):
    if rot45p is None:
        rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    if rot45n is None:
        rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v


def rays_mesh_intersections(rays_from, rays_to, vertices, faces, mode='barycentric', ):
    """calculate the intersections for rays and mesh, where rays_from and rays_to denoted 
    the end points of rays, vertices and faces represented a watertight mesh.

    Args:
        rays_from (ndarray): (N_rays, 3)
        rays_to (ndarray): (N_rays, 3)
        vertices (ndarray): (N_vertices, 3)
        faces (ndarray: int): (N_faces, 3)
        mode (str, optional): _description_. Defaults to 'barycentric'.

    Returns:
        _type_: _description_
    """
    
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
            ray_intersects = pycu.test(vertices, faces, rays_from, rays_to, cfg)
        else:
            # Find intersecting [rays, distances, triangles and points]
            (intersecting_rays, distances, hit_triangles, hit_points) \
                = pycu.test(vertices, faces, rays_from, rays_to, cfg)
    
    if mode!='barycentric':
        return ray_intersects
    else:
        return intersecting_rays, distances, hit_triangles, hit_points


def rays_mesh_intersections_pcu(rays_o, rays_d, vertices, faces, mode='barycentric', ):
    faces_id, barycentric, distances = pcu.ray_mesh_intersection(
        vertices.astype(np.float32), 
        faces.astype(int), 
        rays_o.astype(np.float32), 
        rays_d.astype(np.float32))
    hit_points = pcu.interpolate_barycentric_coords(faces, faces_id, barycentric, vertices)
    hit_mask = faces_id > 0
    hit_triangles = faces_id
    return hit_mask, hit_triangles, hit_points, distances

def inv_transform_points_smpl_verts(self, points, smpl_verts, skinning_weights, bone_transforms, trans, coord_min, coord_max, center):
        ''' Backward skinning based on nearest neighbor SMPL skinning weights '''
        batch_size, n_pts, _ = points.size()
        device = points.device
        knn_ret = pytorch3d.ops.knn_points(points, smpl_verts)
        p_idx = knn_ret.idx.squeeze(-1)
        bv, _ = torch.meshgrid([torch.arange(batch_size).to(device), torch.arange(n_pts).to(device)], indexing='ij')
        pts_W = skinning_weights[bv, p_idx, :]
        # _, part_idx = pts_W.max(-1)

        transforms_fwd = torch.matmul(pts_W, bone_transforms.view(batch_size, -1, 16)).view(batch_size, n_pts, 4, 4)
        transforms_bwd = torch.inverse(transforms_fwd)

        homogen_coord = torch.ones(batch_size, n_pts, 1, dtype=torch.float32, device=device)
        points_homo = torch.cat([points - trans, homogen_coord], dim=-1).view(batch_size, n_pts, 4, 1)
        points_new = torch.matmul(transforms_bwd, points_homo)[:, :, :3, 0]
        points_new = normalize_canonical_points(points_new, coord_min, coord_max, center)

        return points_new, transforms_fwd