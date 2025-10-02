import open3d as o3d
import numpy as np
import freenect
import cv2
from datetime import datetime

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
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False, depth_scale=1000.0, depth_trunc=4.0
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
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic,
        extrinsic=extrinsic
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
        o3d.io.write_point_cloud("final_pointcloud.ply", pcd_total)
    finally:
        freenect.sync_stop()
        vis.destroy_window()
        print("✅ Done. Point cloud saved as final_pointcloud.ply")

if __name__ == "__main__":
    main()
