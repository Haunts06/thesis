import open3d as o3d
import numpy as np
import freenect
import cv2
from datetime import datetime
from pathlib import Path
import os

# === Create folder to save point clouds ===
SAVE_DIR = Path("new_pointclouds")
SAVE_DIR.mkdir(exist_ok=True)

# === Camera intrinsics for Kinect v1 (640x480) ===
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=640,
    height=480,
    fx=525.0,
    fy=525.0,
    cx=319.5,
    cy=239.5
)

def get_depth():
    depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
    if depth is None:
        raise ValueError("Could not get depth frame.")
    return depth.astype(np.uint16)

def get_rgb():
    rgb, _ = freenect.sync_get_video()
    if rgb is None:
        raise ValueError("Could not get RGB frame.")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def get_rgbd_image(rgb, depth):
    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=4.0
    )

def odom_delta(prev_rgbd, curr_rgbd):
    option = o3d.pipelines.odometry.OdometryOption(
        iteration_number_per_pyramid_level=o3d.utility.IntVector([20, 10, 5]),
        depth_diff_max=0.07
    )
    success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
        curr_rgbd, prev_rgbd,
        pinhole_camera_intrinsic,
        np.identity(4),
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option
    )
    return success, trans

def create_point_cloud(rgbd, intrinsic, extrinsic):
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic=extrinsic)

def postprocess_pointcloud(pcd):
    print("🧹 Post-processing point cloud...")
    pcd = pcd.voxel_down_sample(voxel_size=0.005)

    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    return pcd

def main():
    vis = o3d.visualization.Visualizer()
    vis.create_window("Live Kinect RGBD Mapping", width=960, height=540)
    geom_added = False
    pcd_total = o3d.geometry.PointCloud()

    prev_rgbd = None
    pose = np.identity(4)

    print("⏳ Capturing... Press Ctrl+C to stop.")

    try:
        while True:
            rgb = get_rgb()
            depth = get_depth()
            rgbd = get_rgbd_image(rgb, depth)

            if prev_rgbd is None:
                prev_rgbd = rgbd
                pcd = create_point_cloud(rgbd, pinhole_camera_intrinsic, pose)
                pcd_total += pcd
            else:
                success, delta = odom_delta(prev_rgbd, rgbd)
                if success:
                    pose = pose @ delta
                    pcd = create_point_cloud(rgbd, pinhole_camera_intrinsic, pose)
                    pcd_total += pcd
                    prev_rgbd = rgbd

            if not geom_added:
                vis.add_geometry(pcd_total)
                geom_added = True
            else:
                vis.update_geometry(pcd_total)
                vis.poll_events()
                vis.update_renderer()

    except KeyboardInterrupt:
        print("\n💾 Saving final point cloud...")

        # Post-process
        pcd_processed = postprocess_pointcloud(pcd_total)

        # Timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = SAVE_DIR / f"pointcloud_{timestamp}.ply"

        o3d.io.write_point_cloud(str(filename), pcd_processed)
        print(f"✅ Done. Saved to {filename}")

    finally:
        freenect.sync_stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()
