
import freenect
import numpy as np
import matplotlib.pyplot as plt
import time

def get_depth():
    depth, _ = freenect.sync_get_depth()
    return depth.astype(np.uint16)

def measure_center_distance(depth, cx=320, cy=240, window=50):
    center_crop = depth[cy-window:cy+window, cx-window:cx+window].astype(np.float32)
    center_crop = center_crop[center_crop > 0]  # exclude zeros
    if len(center_crop) == 0:
        return None
    return np.mean(center_crop) / 1000.0  # mm to meters

readings = []
timestamps = []

print("Starting repeatability test at center...")

for i in range(20):
    depth = get_depth()
    dist = measure_center_distance(depth)
    if dist is not None:
        readings.append(dist)
        timestamps.append(time.time())
        print(f"[{i+1}] Distance: {dist:.4f} m")
    else:
        print(f"[{i+1}] Invalid reading (no depth data)")

    time.sleep(0.5)  # wait before next capture

readings = np.array(readings)
mean_dist = np.mean(readings)
std_dist = np.std(readings)

print(f"\nRepeatability Test Results:")
print(f"Mean Distance: {mean_dist:.4f} m")
print(f"Standard Deviation: {std_dist:.4f} m")

# Plot the readings
plt.figure(figsize=(10, 4))
plt.plot(readings, marker='o', label='Distance')
plt.axhline(mean_dist, color='green', linestyle='--', label='Mean')
plt.title('Kinect Depth Repeatability Test')
plt.xlabel('Sample Index')
plt.ylabel('Distance (m)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()