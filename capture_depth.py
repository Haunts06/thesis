import freenect, cv2
import numpy as np

def capture_depth(filename="depth.png"):
    depth_mm, _ = freenect.sync_get_depth(format=freenect.DEPTH_MM)  # <-- mm, not 11-bit index
    if depth_mm is None:
        raise RuntimeError("No depth frame captured")
    cv2.imwrite(filename, depth_mm.astype(np.uint16))  # 16-bit PNG
    print(f"[OK] Saved Depth (mm) to {filename}")
    return filename
