import numpy as np
import pickle
import torch
import os
import sys
sys.path.append(".")
import torch.nn.functional as F

from libs.utils.network_utils import MotionBasisComputer
from libs.utils.body_utils import body_pose_to_body_RTs, get_canonical_global_tfms


def main():
    # pose = torch.from_numpy(pose).reshape(-1 ,3).float()
    # tpose_joints = torch.from_numpy(tpose_joints).float()

    model = MotionBasisComputer(total_bones=24)
    Rs, Ts = body_pose_to_body_RTs(pose.reshape(-1, 3), tpose_joints)
    cnl_gtfms = get_canonical_global_tfms(tpose_joints)

    Rs = torch.from_numpy(Rs).float()[None, ...]
    Ts = torch.from_numpy(Ts).float()[None, ...]
    cnl_gtfms = torch.from_numpy(cnl_gtfms).float()[None, ...]
    dst_gtfms=model(Rs, Ts, cnl_gtfms)
    print(dst_gtfms)




def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    batch_size = rot_mats.shape[0]
    num_joints = joints.shape[1]
    device = rot_mats.device

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = torch.cat([joints, torch.zeros([batch_size, num_joints, 1, 1], dtype=dtype, device=device)],dim=2)
    init_bone = torch.matmul(transforms, joints_homogen)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    rel_transforms = transforms - init_bone

    return posed_joints, rel_transforms, transforms

def main1(pose, tpose_joints, SMPL_PARENT):
    Rs, Ts = body_pose_to_body_RTs(pose.reshape(-1, 3), tpose_joints)
    Rs = torch.from_numpy(Rs).float()[None, ...]
    tpose_joints = torch.from_numpy(tpose_joints).float()[None, ...]
    
    posed_joints, rel_transforms, transforms = batch_rigid_transform(Rs, tpose_joints, SMPL_PARENT)
    print()
    
if __name__=="__main__":
    SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}
    parents = np.array([-1]+list(SMPL_PARENT.values()))

    # pose
    bones_tfms_path = "data/data_prepared/CoreView_377_arah/models/000000.npz"
    data = np.load(bones_tfms_path)
    root_orient = data['root_orient']
    pose_body = data['pose_body']
    pose_hand = data['pose_hand']
    pose = np.concatenate([root_orient, pose_body, pose_hand])

    # tpose joints
    tpose_path = "data/data_prepared/CoreView_377/0/canonical_joints.pkl"
    with open(tpose_path, "rb") as f:
        data = pickle.load(f)
    tpose_joints = data['joints']
    
    main1(pose, tpose_joints, parents)