import freenect
import numpy as np
import cv2

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth.astype(np.uint16)

def estimate_distance_accuracy(depth_image, cx=320, cy=240, window=25):
    center_crop = depth_image[cy-window:cy+window, cx-window:cx+window]
    z_values = center_crop.astype(np.float32)
    z_values = z_values[z_values > 0]  # remove zeros
    z_meters = z_values / 1000.0  # convert mm → meters
    mean = np.mean(z_meters)
    std = np.std(z_meters)
    return mean, std


def show_depth_with_box(depth_image, cx=320, cy=240, window=100):
    vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    vis = vis.astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (cx - window, cy - window), (cx + window, cy + window), (0, 255, 0), 2)
    cv2.imshow("Depth View (center box = measurement area)", vis)

if __name__ == "__main__":
    print("Capturing depth... Press ESC to exit.")

    while True:
        depth = get_depth()
        mean_dist, std_dev = estimate_distance_accuracy(depth)
        print(f"Distance = {mean_dist:.3f} m | Std Dev = {std_dev:.3f} m", end="\r")

        show_depth_with_box(depth)
        key = cv2.waitKey(30)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()
