import freenect, cv2

def capture_rgb(filename="rgb.png") -> str:
    rgb, _ = freenect.sync_get_video()
    if rgb is None:
        raise RuntimeError("No RGB frame captured")

    cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[OK] Saved RGB to {filename}")
    return filename  # <-- always return the path
