import json, pickle
import os,cv2
import numpy as np
import sys
sys.path.append('.')

from tools.smpl.smpl_numpy import SMPL

i_frame=0
def apply_global_tfm_to_camera(E, Rh, Th):
    r""" Get camera extrinsics that considers global transformation.

    Args:
        - E: Array (3, 3)
        - Rh: Array (3, )
        - Th: Array (3, )
        
    Returns:
        - Array (3, 3)
    """

    global_tfms = np.eye(4)  #(4, 4)
    global_rot = cv2.Rodrigues(Rh)[0].T
    global_trans = Th
    global_tfms[:3, :3] = global_rot
    global_tfms[:3, 3] = -global_rot.dot(global_trans)
    return E.dot(np.linalg.inv(global_tfms))

def construct_camera_matrix(input_filapath, output_filepath):
    # poses and shapes
    params_path="data/CoreView_377/new_params/0.npy"
    data=np.load(params_path,allow_pickle=True).item()
    poses=data['poses']
    shapes=data['shapes']
    Rh,Th=data['Rh'][i_frame],data['Th'][i_frame]
    
    anns=np.load(input_filapath,allow_pickle=True).item()
    Ks=anns['cams']['K']
    Ds=anns['cams']['D']
    Rs=anns['cams']['R']
    Ts=anns['cams']['T']

    cameras_sphere={}

    for i in range(len(Rs)):
        # extrinsic matrix
        E_inv=np.eye(4)
        E_inv[:3,:3]=Rs[i]
        E_inv[:3,3:]=Ts[i]/1000 # scale unit to pixel
        E_inv = apply_global_tfm_to_camera(
                E=E_inv, 
                Rh=Rh,
                Th=Th)
        E = np.linalg.inv(E_inv)
        E = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]) @ E
        E = np.linalg.inv(E) ## 377
        
        # fov
        K=np.eye(4)
        K[:3,:3]=Ks[i] # (3, 3)
        P=K@E
        S=np.eye(4)
        cameras_sphere[f'scale_mat_{i}']= S 
        cameras_sphere[f'world_mat_{i}']= P

    np.savez(output_filepath, **cameras_sphere)
    
def construct_images_masks(images_dir_in, masks_dir_in, images_dir, masks_dir):
    # bg_rgb=np.array([0,0,0])
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    nframe=f'00{i_frame}'
    cameras=range(1,24)

    for c in cameras:
        c_image=cv2.imread(os.path.join(images_dir_in,f"Camera_B{c}/000{nframe}.jpg"))
        c_mask=cv2.imread(os.path.join(masks_dir_in,f"Camera_B{c}/000{nframe}.png"))
        new_image = c_image * c_mask
        
        # c_mask = np.ones_like(c_image,dtype=np.uint8)
        cv2.imwrite(os.path.join(images_dir,f"{str(c).zfill(3)}.png"),new_image)
        cv2.imwrite(os.path.join(masks_dir,f"{str(c).zfill(3)}.png"),c_mask*255)
        
    new_image.shape
    return

def main1():
    """_summary_
    construct a mini dataset from CoreView_377 with multi-view images
    """
    annots_filapath='/home/wanghao/workspace/gaussian_human/data/CoreView_377/annots.npy'
    images_filepath='data/CoreView_377'
    masks_filepath='data/CoreView_377/mask'
    
    dirpath='data/377-synthetic'
    os.makedirs(dirpath, exist_ok=True)
    cameras_filepath='cameras_sphere.npz'
    images_dir='image'
    masks_dir='mask'
    
    construct_camera_matrix(annots_filapath, os.path.join(dirpath, cameras_filepath))
    construct_images_masks(images_filepath, masks_filepath, os.path.join(dirpath, images_dir), os.path.join(dirpath, masks_dir))

def main2():
    """_summary_
    """
    # mesh_infos_path = 'data/data_prepared/377/0/mesh_infos.pkl' # scale
    mesh_infos_path = "data/data_prepared/377_unscale/mesh_infos.pkl" # unscale
    with open(mesh_infos_path, 'rb') as f:
        mesh_infos = pickle.load(f)
    
    tpose_joints= mesh_infos['frame_000001']['tpose_joints']
    
    joints_all=[]
    for key in mesh_infos.keys():
        joints_all.append(mesh_infos[key]['joints'])
    joints_all = np.concatenate(joints_all,axis=0)
    print()
    
    
def construct_camera_pkl(annots_filapath, camera_pkl):
    annots = np.load(annots_filapath, allow_pickle=True).item()
    view_i = 0
    iname = annots['ims'][view_i]['ims'][0][-10:]
    cameras_name = [ipath[:-11] for ipath in annots['ims'][0]['ims']]
    Ks = annots['cams']['K']
    Ds = annots['cams']['D']
    Rs = annots['cams']['R']
    Ts = annots['cams']['T']
    camera_data={}
    for i, camera in enumerate(cameras_name):
        E=np.eye(4)
        E[:3,:3]=Rs[i]
        E[:3,3:]=Ts[i]
        D=Ds[i]
        K=Ks[i]
        camera_data[camera+iname[:-4]]={
            'intrinsics': K,
            'extrinsics': E,
            'distortions': D.squeeze()
        }
        
    # write camera infos
    with open(camera_pkl, 'wb') as f:   
        pickle.dump(camera_data, f)

def construct_mesh_infos_pkl(annots_filapath, params_filapath, mesh_infos_pkl):
    sex="neutral"
    MODEL_DIR='tools/smpl/models/'
    
    annots = np.load(annots_filapath, allow_pickle=True).item()
    view_i = 0
    iname = annots['ims'][view_i]['ims'][0][-10:]
    cameras_name = [ipath[:-11] for ipath in annots['ims'][0]['ims']]
    
    # load smpl parameters
    smpl_params = np.load(params_filapath, allow_pickle=True).item()

    betas = smpl_params['shapes'][0] #(10,)
    poses = smpl_params['poses'][0]  #(72,)
    Rh = smpl_params['Rh'][0]  #(3,)
    Th = smpl_params['Th'][0]  #(3,)
    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
    # write mesh info
    _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
    _, joints = smpl_model(poses, betas)
    
    mesh_infos={}
    for i, camera in enumerate(cameras_name):
        mesh_infos[camera+iname[:-4]]={
            'Rh':Rh,
            'Th':Th,
            'poses':poses,
            'beats':betas,
            'joints':joints,
            'tpose_joints':tpose_joints
        }
    
    # write camera infos
    with open(mesh_infos_pkl, 'wb') as f:   
        pickle.dump(mesh_infos, f)

def construct_images_masks2(annots_filapath, images_dir_in, masks_dir_in, images_dir, masks_dir):
    annots = np.load(annots_filapath, allow_pickle=True).item()
    view_i = 0
    iname = annots['ims'][view_i]['ims'][0][-10:]
    cameras_name = [ipath[:-11] for ipath in annots['ims'][0]['ims']]

    # bg_rgb=np.array([0,0,0])
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    nframe=f'00{i_frame}'
    cameras=range(1,24)

    # for c in cameras:
    for i, camera in enumerate(cameras_name):
        c_image=cv2.imread(os.path.join(images_dir_in,f"Camera_B{i+1}/{iname[:-4]}.jpg"))
        c_mask=cv2.imread(os.path.join(masks_dir_in,f"Camera_B{i+1}/{iname[:-4]}.png"))
        new_image = c_image * c_mask
        
        cv2.imwrite(os.path.join(images_dir,f"{camera+iname[:-4]}.png"),new_image)
        cv2.imwrite(os.path.join(masks_dir,f"{camera+iname[:-4]}.png"),c_mask*255)
        
    return

def main3():
    """_summary_
    construct a mini dataset from CoreView_377 with multi-view images
    """
    annots_filapath='/home/wanghao/workspace/gaussian_human/data/CoreView_377/annots.npy'
    images_filepath='data/CoreView_377'
    masks_filepath='data/CoreView_377/mask'
    
    dirpath='data/data_prepared/377_m'
    os.makedirs(dirpath, exist_ok=True)
    camera_pkl='cameras.pkl'
    mesh_info_pkl='mesh_infos.pkl'
    images_dir='image'
    masks_dir='mask'
    # construct_camera_pkl(annots_filapath, os.path.join(dirpath, camera_pkl))
    # construct_mesh_infos_pkl(annots_filapath, "data/CoreView_377/new_params/0.npy", os.path.join(dirpath, mesh_info_pkl))
    construct_images_masks2(annots_filapath, images_filepath, masks_filepath, os.path.join(dirpath, images_dir), os.path.join(dirpath, masks_dir))
    
    # construct_images_masks(images_filepath, masks_filepath, os.path.join(dirpath, images_dir), os.path.join(dirpath, masks_dir))
    return

def main4():
    def load_K_Rt_from_P(filename, P=None):
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose
    
    n_images = 23
    camera_dict = np.load("data/377-synthetic/cameras_sphere.npz")
    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    scale_mats_np = []

    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []

    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        # pose = np.linalg.inv(pose) ## 377
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)

if __name__=="__main__":
    main1()
