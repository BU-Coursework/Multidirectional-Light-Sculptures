import os
import numpy as np
import trimesh

def mesh_from_voxels(voxels: np.ndarray, pitch: float):
    if voxels.sum() == 0:
        raise ValueError("Voxel grid is empty; cannot export mesh.")
    return trimesh.voxel.ops.matrix_to_marching_cubes(voxels, pitch=pitch)


def export_voxels_to_stl(voxels: np.ndarray, pitch: float, out_path: str):
    mesh = mesh_from_voxels(voxels, pitch)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mesh.export(out_path)
    return mesh


def mesh_triangles_in_world(mesh, world_size, pitch):
    # Account for the STL's float32 precision and marching-cubes index origin.
    return mesh.triangles.astype(np.float32).astype(float) + (pitch - world_size) / 2
