import cv2
import freenect
import numpy as np
from capture_rgb import capture_rgb
from capture_depth import capture_depth
from make_pointcloud import make_pointcloud
from datetime import datetime
from pathlib import Path

OUT = Path("captures")
OUT.mkdir(exist_ok=True)

def unique_names():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (OUT / f"rgb_{ts}.png",
            OUT / f"depth_{ts}.png",
            OUT / f"pointcloud_{ts}.ply")

def show_live_preview():
    rgb, _ = freenect.sync_get_video(format=freenect.VIDEO_RGB)
    depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)

    if rgb is None or depth is None:
        return

    # Convert for display
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_vis = cv2.convertScaleAbs(depth, alpha=255.0/np.max(depth))
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("RGB", rgb_bgr)
    cv2.imshow("Depth", depth_vis)

if __name__ == "__main__":
    print("RGBD capture loop.")
    print(" - Live preview shown in OpenCV windows")
    print(" - Press Enter in terminal to capture, type 'q' + Enter to quit")
    try:
        while True:
            # show live preview until user presses Enter
            while True:
                show_live_preview()
                key = cv2.waitKey(1) & 0xFF
                # escape preview if user presses 'c' in the preview window
                if key == ord("c"):
                    break
                # exit whole program if user presses 'q' in preview window
                if key == ord("q"):
                    raise KeyboardInterrupt

                # non-blocking check: see if user pressed Enter in terminal
                import sys, select
                if select.select([sys.stdin], [], [], 0)[0]:
                    cmd = sys.stdin.readline().strip().lower()
                    if cmd == "q":
                        raise KeyboardInterrupt
                    else:
                        break

            rgb_path, depth_path, ply_path = unique_names()
            try:
                rgb_file = capture_rgb(str(rgb_path))
                depth_file = capture_depth(str(depth_path))
                make_pointcloud(rgb_file, depth_file, str(ply_path), show=True)
            except Exception as e:
                print(f"[ERR] {e}")
    except KeyboardInterrupt:
        pass
    finally:
        freenect.sync_stop()
        cv2.destroyAllWindows()
        print("Byeeee 👋")
