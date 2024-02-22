import json
import numpy as np
import cv2 

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

# scale_mat_0 = np.array([[231.0682, 0, 0, -28.461128],[0, 231.0682, 0, -13.186272],[0,0, 231.0682, 641.4886],[0, 0,0,1]])
# scale_mat_inv_0 = np.linalg.inv(scale_mat_0)


# dtu_scan24_transform="data/dtu_scan24/transform.json"
# with open(dtu_scan24_transform, 'r') as f:
#     transforms = json.load(f)

# frames = transforms['frames']
# int_mat = np.array(frames[0]['intrinsic_matrix'])
# ext_mat = np.array(frames[0]['transform_matrix'])

# world_mat = int_mat @ np.linalg.inv(ext_mat)
# print(transforms)

cameras_sphere_file = "data/dtu_scan24/cameras_sphere.npz"
data = np.load(cameras_sphere_file, allow_pickle=True)
scale_mat_0 = data['scale_mat_0']
world_mat_0 = data['world_mat_0']
camera_mat_0 = data['camera_mat_0']

P = world_mat_0 @ scale_mat_0
int_mat, ext_mat = load_K_Rt_from_P(None, P[:3, :4])
print()