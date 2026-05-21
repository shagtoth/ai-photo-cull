from pathlib import Path
import cv2
import rawpy

SUPPORTED_RAW = [".cr2", ".dng", ".nef", ".arw", ".orf", ".rw2"]
SUPPORTED_IMG = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_RAW + SUPPORTED_IMG


def extract_preview(path: Path):
    ext = path.suffix.lower()
    if ext in SUPPORTED_IMG:
        return cv2.imread(str(path))

    if ext in SUPPORTED_RAW:
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return None
