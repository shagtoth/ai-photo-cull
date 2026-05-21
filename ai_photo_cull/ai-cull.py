#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rawpy
from PIL import Image

SUPPORTED_RAW = [".cr2", ".dng", ".nef", ".arw", ".orf", ".rw2"]
SUPPORTED_IMG = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

YOLO_MODEL = "models/yolov8n.onnx"
AESTHETIC_MODEL = "models/laion_aesthetic.onnx"
FACE_MODEL = "models/face_detector.onnx"

HTML_REPORT_NAME = "ai_cull_report.html"


# -----------------------------
# MODEL LOADING (GPU-AWARE)
# -----------------------------

def load_onnx(model_path):
    providers = []
    # Try CoreML / Metal first (Apple Silicon), then CPU
    for p in [
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]:
        if p in ort.get_available_providers():
            providers.append(p)
    if not providers:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


yolo = load_onnx(YOLO_MODEL)
aesthetic = load_onnx(AESTHETIC_MODEL)
face_det = load_onnx(FACE_MODEL)


# -----------------------------
# IMAGE LOADING
# -----------------------------

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


# -----------------------------
# BASIC METRICS
# -----------------------------

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def exposure_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    blacks = hist[0]
    whites = hist[-1]
    clipped = (blacks + whites) / hist.sum()
    return float(1.0 - clipped)


def noise_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.Laplacian(blur, cv2.CV_64F).var()
    return float(min(noise / 500.0, 1.0))


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    high_freq = np.sum(magnitude > np.percentile(magnitude, 95))
    return float(min(high_freq / 50000.0, 1.0))


# -----------------------------
# AI HELPERS
# -----------------------------

def detect_subjects(image):
    img = cv2.resize(image, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]
    outputs = yolo.run(None, {yolo.get_inputs()[0].name: img})[0]
    score = float(np.mean(outputs[:, 4]))
    return max(score, 0.0)


def aesthetic_score(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]
    outputs = aesthetic.run(None, {aesthetic.get_inputs()[0].name: img})[0]
    return float(outputs[0][0])


def face_eye_sharpness(image):
    """Detect faces, then compute sharpness in eye/face region."""
    h, w = image.shape[:2]
    img = cv2.resize(image, (320, 320))
    img_norm = img.astype(np.float32) / 255.0
    img_norm = np.transpose(img_norm, (2, 0, 1))[None, :, :, :]

    outputs = face_det.run(None, {face_det.get_inputs()[0].name: img_norm})[0]
    if outputs.shape[0] == 0:
        return 0.0

    # Assume outputs: [N, 5] -> x1, y1, x2, y2, score (simple face model)
    best = outputs[np.argmax(outputs[:, 4])]
    x1, y1, x2, y2, conf = best
    if conf < 0.3:
        return 0.0

    # Map back to original image
    x1 = int(x1 / 320 * w)
    x2 = int(x2 / 320 * w)
    y1 = int(y1 / 320 * h)
    y2 = int(y2 / 320 * h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    face_roi = image[y1:y2, x1:x2]
    if face_roi.size == 0:
        return 0.0

    sharp = variance_of_laplacian(face_roi)
    return float(min(sharp / 400.0, 1.0))


# -----------------------------
# SCORING PROFILES
# -----------------------------

def score_image(image, profile):
    sharp = variance_of_laplacian(image)
    expo = exposure_score(image)
    noise = noise_score(image)
    blur = blur_score(image)
    subj = detect_subjects(image)
    aesth = aesthetic_score(image)
    face_sharp = face_eye_sharpness(image)

    sharp_norm = min(sharp / 500.0, 1.0)

    if profile == "sports":
        final = (
            0.40 * sharp_norm +
            0.10 * face_sharp +
            0.15 * expo +
            0.10 * noise +
            0.10 * blur +
            0.05 * subj +
            0.10 * aesth
        )
    elif profile == "burlesque":
        final = (
            0.20 * sharp_norm +
            0.20 * face_sharp +
            0.20 * expo +
            0.10 * noise +
            0.05 * blur +
            0.10 * subj +
            0.15 * aesth
        )
    else:  # derby
        final = (
            0.35 * sharp_norm +
            0.15 * face_sharp +
            0.15 * expo +
            0.10 * noise +
            0.10 * blur +
            0.05 * subj +
            0.10 * aesth
        )

    return {
        "final": final,
        "sharp_norm": sharp_norm,
        "expo": expo,
        "noise": noise,
        "blur": blur,
        "subj": subj,
        "aesth": aesth,
        "face_sharp": face_sharp,
    }


# -----------------------------
# METADATA WRITING (XMP)
# -----------------------------

def set_xmp_flag(path, rating=None, label=None, pick=None, reject=None):
    cmd = ["exiftool", "-overwrite_original"]
    if rating is not None:
        cmd.append(f"-XMP:Rating={rating}")
    if label is not None:
        cmd.append(f"-XMP:Label={label}")
    if pick is not None:
        cmd.append(f"-XMP:Pick={pick}")
    if reject is not None:
        cmd.append(f"-XMP:Reject={reject}")
    cmd.append(str(path))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# -----------------------------
# HTML CONTACT SHEET
# -----------------------------

def generate_html_report(results, output_path: Path):
    rows = []
    for r in results:
        path = r["path"]
        s = r["scores"]
        flag = r["flag"]
        color = {
            "TOP": "#c8f7c5",
            "KEEP": "#ffffff",
            "REJECT": "#f7c5c5",
        }.get(flag, "#ffffff")

        rows.append(f"""
<tr style="background:{color}">
  <td><img src="{path.as_posix()}" style="max-width:200px; max-height:150px;"></td>
  <td>{path.name}</td>
  <td>{s['final']:.3f}</td>
  <td>{s['sharp_norm']:.3f}</td>
  <td>{s['face_sharp']:.3f}</td>
  <td>{s['expo']:.3f}</td>
  <td>{s['noise']:.3f}</td>
  <td>{s['blur']:.3f}</td>
  <td>{s['aesth']:.3f}</td>
  <td>{flag}</td>
</tr>
""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI Cull Report</title>
<style>
body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 4px; font-size: 12px; }}
th {{ background: #eee; }}
</style>
</head>
<body>
<h1>AI Cull Report</h1>
<table>
<thead>
<tr>
  <th>Preview</th>
  <th>File</th>
  <th>Final</th>
  <th>Sharp</th>
  <th>Face Sharp</th>
  <th>Exposure</th>
  <th>Noise</th>
  <th>Blur</th>
  <th>Aesthetic</th>
  <th>Flag</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def process_directory(folder, profile):
    folder = Path(folder)
    scored = []

    for path in folder.rglob("*"):
        if path.suffix.lower() not in SUPPORTED_RAW + SUPPORTED_IMG:
            continue

        print(f"Processing {path}...")
        img = extract_preview(path)
        if img is None:
            print("  Could not load preview.")
            continue

        scores = score_image(img, profile)

        auto_reject = (
            scores["sharp_norm"] < 0.15 or
            scores["expo"] < 0.2 or
            scores["blur"] < 0.1
        )

        if auto_reject:
            set_xmp_flag(path, reject=1, label="Red", rating=1)
            print("  → AUTO REJECT")
            scored.append({"path": path, "scores": scores, "flag": "REJECT"})
        else:
            scored.append({"path": path, "scores": scores, "flag": "KEEP"})

    # Rank non-rejected
    keepers = [r for r in scored if r["flag"] != "REJECT"]
    keepers.sort(key=lambda r: r["scores"]["final"], reverse=True)

    if keepers:
        top_n = max(1, len(keepers) // 10)
        for i, r in enumerate(keepers):
            path = r["path"]
            if i < top_n:
                set_xmp_flag(path, pick=1, label="Green", rating=5)
                r["flag"] = "TOP"
                print(f"TOP PICK → {path.name}")
            else:
                set_xmp_flag(path, rating=3)

    # HTML report
    report_path = folder / HTML_REPORT_NAME
    generate_html_report(scored, report_path)
    print(f"HTML report written to: {report_path}")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Image Culling Tool")
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument(
        "--profile",
        choices=["sports", "burlesque", "derby"],
        default="sports",
        help="Scoring profile",
    )
    args = parser.parse_args()
    process_directory(args.folder, args.profile)


if __name__ == "__main__":
    main()

