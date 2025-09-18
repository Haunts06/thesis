import os
from pathlib import Path
import open3d as o3d

def make_pointcloud(rgb_file="rgb.png", depth_file="depth.png", out_file="pointcloud.ply", show=True):
    rgb_path   = Path(rgb_file)
    depth_path = Path(depth_file)
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    # Read images (must pass str/Path-like)
    color = o3d.io.read_image(os.fspath(rgb_path))
    depth = o3d.io.read_image(os.fspath(depth_path))

    # Kinect v1 intrinsics (640x480)
    intr = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )

    # Depth is in millimeters -> depth_scale=1000.0
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=1000.0,   # mm -> meters
        depth_trunc=4.0,      # ignore > 3m
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    # Flip to a conventional orientation
    pcd.transform([[1,0,0,0],
                   [0,-1,0,0],
                   [0,0,-1,0],
                   [0,0,0,1]])

    o3d.io.write_point_cloud(os.fspath(out_file), pcd)
    print(f"[OK] Saved point cloud to {out_file}")

    if show:
        o3d.visualization.draw_geometries([pcd])
