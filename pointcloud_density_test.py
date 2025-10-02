
import freenect
import numpy as np
import open3d as o3d

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth.astype(np.uint16)

def create_point_cloud(depth):
    fx, fy = 594.21, 591.04
    cx, cy = 339.5, 242.7
    rows, cols = depth.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    z = depth / 1000.0
    x3 = (x - cx) * z / fx
    y3 = (y - cy) * z / fy
    xyz = np.stack((x3, y3, z), axis=-1).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def compute_density(pcd, voxel_size=0.01):
    voxel_grid = pcd.voxel_down_sample(voxel_size=voxel_size)
    total_volume_m3 = voxel_size ** 3 * len(voxel_grid.points)
    point_density = len(pcd.points) / total_volume_m3 if total_volume_m3 > 0 else 0
    print(f"Original points       : {len(pcd.points)}")
    print(f"Voxel grid points     : {len(voxel_grid.points)}")
    print(f"Voxel size            : {voxel_size} m")
    print(f"Estimated volume (m³) : {total_volume_m3:.4f}")
    print(f"Estimated density     : {point_density:.2f} points/m³")
    return voxel_grid

if __name__ == "__main__":
    depth = get_depth()
    pcd = create_point_cloud(depth)
    voxel_grid = compute_density(pcd, voxel_size=0.01)
    voxel_grid.paint_uniform_color([0.2, 0.6, 1.0])
    o3d.visualization.draw_geometries([voxel_grid])
