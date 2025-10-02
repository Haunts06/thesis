
import freenect
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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

def analyze_planarity(pcd, distance_threshold=0.005):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    [a, b, c, d] = plane_model
    inlier_cloud = pcd.select_by_index(inliers)
    distances = np.abs(np.asarray(inlier_cloud.points) @ np.array([a, b, c]) + d)
    mean_dev = np.mean(distances)
    std_dev = np.std(distances)
    print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"Inlier points: {len(inlier_cloud.points)}")
    print(f"Mean deviation from plane: {mean_dev:.6f} m")
    print(f"Standard deviation from plane: {std_dev:.6f} m")
    return inlier_cloud, plane_model, distances

if __name__ == "__main__":
    depth = get_depth()
    pcd = create_point_cloud(depth)
    print("Fitting plane to flat surface...")
    inlier_cloud, plane_model, distances = analyze_planarity(pcd)

    inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([inlier_cloud])

    # Histogram of distances to plane
    plt.hist(distances * 1000, bins=50)  # convert to mm
    plt.title("Deviation from Fitted Plane")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Number of Points")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
