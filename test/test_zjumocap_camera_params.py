import json, pickle
import os,cv2
import numpy as np
import sys
sys.path.append('.')

################################ TOOLS ###################
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
################################ TOOLS ###################


def dtu_to_neus2(data_dir, cameras, outdata_dir):
    # cameras split
    split_cameras={
        'all': cameras,
        'test': cameras[::6],
        'train': [c for c in cameras if c not in cameras[::6]]
    }
    
    cameras_sphere_file = os.path.join(data_dir, "cameras_sphere.npz")
    data = np.load(cameras_sphere_file, allow_pickle=True)
    transforms = {
        "w": 1024, 
        "h": 1024,
        "aabb_scale": 1.0,
        "scale": 0.5,
        "offset": [0.5, 0.5, 0.5],
        "from_na": True,
        "frames": []
    }
    
    # write images
    images_dir = os.path.join(outdata_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for camera in cameras:
        image_file = os.path.join(data_dir, "image", f"{camera-1}".zfill(6)+".png")
        mask_file = os.path.join(data_dir, "mask", f"{camera-1}".zfill(3)+".png")
        mask = cv2.imread(mask_file)
        image = cv2.imread(image_file)
        image_alpha = np.concatenate([image, mask[..., -1:]], axis=-1)
        cv2.imwrite(os.path.join(images_dir, str(camera-1).zfill(6)+".png"), image_alpha)
    
    for split, cameras in split_cameras.items():
        for camera in cameras:
            scale_mat = data[f'scale_mat_{camera-1}']
            world_mat = data[f'world_mat_{camera-1}']
            camera_mat = data[f'camera_mat_{camera-1}']

            P = world_mat @ scale_mat
            int_mat, ext_mat = load_K_Rt_from_P(None, P[:3, :4])
            frame = {
                "file_path": f"images/{str(camera-1).zfill(6)}.png",
                "transform_matrix": ext_mat.tolist(),
                "intrinsic_matrix": int_mat.tolist(),
            }
            transforms['frames'].append(frame)

        transforms_file = os.path.join(outdata_dir, f"transform_{split}.json")
        with open(transforms_file, "w") as f:
            json.dump(transforms, f)
            
def to_dtu_format(data_dir, frame_id, cameras, outdata_dir):
    # write images
    images_dir = os.path.join(outdata_dir, "image")
    masks_dir = os.path.join(outdata_dir, "mask")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    for camera in cameras:
        image_file = os.path.join(data_dir, f"Camera_B{camera}", f"{frame_id}".zfill(6)+".jpg")
        mask_file = os.path.join(data_dir, "mask", f"Camera_B{camera}", f"{frame_id}".zfill(6)+".png")
        mask = cv2.imread(mask_file)
        image = cv2.imread(image_file)
        cv2.imwrite(os.path.join(images_dir, f"{camera-1}".zfill(6)+".png"), image)
        cv2.imwrite(os.path.join(masks_dir, f"{camera-1}".zfill(3)+".png"), mask*255)
    
    # load ext and ixt params
    annots_npy_file = os.path.join(data_dir, "annots.npy")
    annots_npy = np.load(annots_npy_file, allow_pickle=True).item()
    Ks = annots_npy['cams']['K']
    Ds = annots_npy['cams']['D']
    Rs = annots_npy['cams']['R']
    Ts = annots_npy['cams']['T']
    
    # load poses and shapes
    params_file = os.path.join(data_dir, f"new_params/{frame_id}.npy")
    params_npy = np.load(params_file,allow_pickle=True).item()
    shapes = params_npy['shapes'] # (1, 10)
    poses = params_npy['poses'] # (1, 72)
    Rh = params_npy['Rh'][0] # (3,)
    Th = params_npy['Th'][0] # (3,)
    
    # save cameras sphere.npz
    cameras_sphere = {}
    for camera in cameras:
        camera = camera - 1
        # camera_mat
        camera_mat_inv = np.eye(4, 4)
        camera_mat_inv[0,0] = 1024 // 2
        camera_mat_inv[0,2] = 1024 // 2
        camera_mat_inv[1,1] = 1024 // 2
        camera_mat_inv[1,2] = 1024 // 2
        
        # world_mat
        E = np.eye(4, 4)
        E[:3, :3] = Rs[camera]
        E[:3, 3:] = Ts[camera] / 1000
        E = apply_global_tfm_to_camera(
            E=E,
            Rh=Rh,
            Th=Th)
        E_inv = np.linalg.inv(E)
        E_inv = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]) @ E_inv # opencv -> dtu
        E = np.linalg.inv(E_inv)
        K = np.eye(4)
        K[:3,:3] = Ks[camera]
        world_mat = K @ E
        
        # scale_mat
        scale_mat_inv = np.eye(4, 4)
        
        cameras_sphere.update({
            f'scale_mat_{camera}': np.linalg.inv(scale_mat_inv),
            f'scale_mat_inv_{camera}': scale_mat_inv,
            f'world_mat_{camera}': world_mat,
            f'world_mat_inv_{camera}': np.linalg.inv(world_mat),
            f'camera_mat_{camera}': np.linalg.inv(camera_mat_inv),
            f'camera_mat_inv_{camera}': camera_mat_inv,
        })
    np.savez(os.path.join(outdata_dir,"cameras_sphere.npz"), **cameras_sphere)
    
def to_blender_format(data_dir, frame_id, cameras, outdata_dir):
    # cameras split
    split_cameras={
        'test': cameras[::6],
        'val': cameras[::6],
        'train': [c for c in cameras if c not in cameras[::6]]
    }
    
    # write images
    for split, cameras in split_cameras.items():
        split_images_dir = os.path.join(outdata_dir, split)
        os.makedirs(split_images_dir, exist_ok=True)
        for camera in cameras:
            image_file = os.path.join(data_dir, f"Camera_B{camera}", f"{frame_id}".zfill(6)+".jpg")
            mask_file = os.path.join(data_dir, "mask", f"Camera_B{camera}", f"{frame_id}".zfill(6)+".png")
            mask = cv2.imread(mask_file)
            image = cv2.imread(image_file) 
            image_alpha = np.concatenate([image, mask[..., -1:]*255], axis=-1)
            cv2.imwrite(os.path.join(split_images_dir, str(camera)+".png"), image_alpha)
    
    # load ext and ixt params
    annots_npy_file = os.path.join(data_dir, "annots.npy")
    annots_npy = np.load(annots_npy_file, allow_pickle=True).item()
    Ks = annots_npy['cams']['K']
    Ds = annots_npy['cams']['D']
    Rs = annots_npy['cams']['R']
    Ts = annots_npy['cams']['T']
    
    # calculate fov
    H, W, _ = image.shape
    mean_K = np.array([K for K in Ks]).sum(0)/len(Ks)
    camera_angle_x = np.arctan(W/2/mean_K[0,0])*2
    camera_angle_y = np.arctan(H/2/mean_K[1,1])*2
    
    # save transforms json
    for split, cameras in split_cameras.items():
        split_transforms_json = os.path.join(outdata_dir, f"transforms_{split}.json")
        transforms_dict = {
            "camera_angle_x": camera_angle_x,
            "camera_angle_y": camera_angle_y,
            "frames": [],
        }
        for camera in cameras:
            R = Rs[camera - 1]
            T = Ts[camera - 1]
            E = np.eye(4,4)
            E[:3, :3] = R # @ np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])
            E[:3, 3:] = T/1000
            E = np.linalg.inv(E)
            E[:3, :3] = E[:3, :3] @ np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])
            
            frame_dict = {
                "file_path": f"./{split}/{camera}",
                "rotation": 0.012566370614359171,
                "transform_matrix": E.tolist()
            }
            transforms_dict['frames'].append(frame_dict)
        
        with open(split_transforms_json, "w") as f:
            json.dump(transforms_dict, f)
    print("Complete")

if __name__=="__main__":
    mode = "dtu2neus2"
    
    if mode == "zjumocap2dtu":
        subject = '377'
        frame_id = 0
        data_dir = f'data/CoreView_{subject}'
        cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        outdata_dir = f"data/data_prepared/mini_zjumocap_{subject}_mvs_dtu"
        to_dtu_format(data_dir, frame_id, cameras, outdata_dir)
    
    if mode == "zjumocap2blender":
        subject = '377'
        frame_id = 0
        data_dir = f'data/CoreView_{subject}'
        cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        outdata_dir = f"data/data_prepared/mini_zjumocap_{subject}_mvs_dtu"
        to_blender_format(data_dir, frame_id, cameras, outdata_dir)
    
    if mode == "dtu2neus2":
        subject = '377'
        data_dir = "data/data_prepared/mini_zjumocap_377_mvs_dtu"
        cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        outdata_dir = f"data/data_prepared/mini_zjumocap_{subject}_mvs_neus2"
        dtu_to_neus2(data_dir, cameras, outdata_dir)
