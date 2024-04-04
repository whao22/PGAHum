import trimesh
import numpy as np


def compute_cd(src_mesh, tgt_mesh, num_samples=1000):
    src_surf_pts, _ = trimesh.sample.sample_surface(
            src_mesh, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(
        tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(
        tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(
        src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist = src_tgt_dist.mean()
    tgt_src_dist = tgt_src_dist.mean()

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    return chamfer_dist

if __name__ == '__main__':
    mesh1_file = "test/eval_cd/data/377_pet-neus_gt.ply"
    mesh2_file = "test/eval_cd/data/377_arah.ply"
    mesh1 = trimesh.load_mesh(mesh1_file)
    mesh2 = trimesh.load_mesh(mesh2_file)
    chamfer_dist = compute_cd(mesh1, mesh2)
    print("chamfer_dist: ", chamfer_dist)