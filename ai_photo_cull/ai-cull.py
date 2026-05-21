#!/usr/bin/env python3
import argparse
import os
import subprocess
import rawpy
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort

# -----------------------------
# CONFIG
# -----------------------------

SUPPORTED_RAW = [".cr2", ".dng", ".nef", ".arw", ".orf", ".rw2"]
SUPPORTED_IMG = [".jpg", ".jpeg", ".png", ".tif"]

# ONNX models (place these in a /models folder)
YOLO_MODEL = "models/yolov8n.onnx"
AESTHETIC_MODEL = "models/laion_aesthetic.onnx"

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def extract_preview(path):
    """Extract JPEG preview from RAW or load JPEG directly."""
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


def variance_of_laplacian(image):
    """Sharpness metric."""
    return cv2.Laplacian(image, cv2.CV_64F).var()


def exposure_score(image):
    """Simple exposure score based on histogram clipping."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    blacks = hist[0]
    whites = hist[-1]
    clipped = (blacks + whites) / hist.sum()
    return float(1.0 - clipped)


def noise_score(image):
    """Estimate noise via Laplacian variance of blurred image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.Laplacian(blur, cv2.CV_64F).var()
    return float(min(noise / 500.0, 1.0))


def blur_score(image):
    """FFT-based blur detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    high_freq = np.sum(magnitude > np.percentile(magnitude, 95))
    return float(min(high_freq / 50000.0, 1.0))


# -----------------------------
# AI MODELS
# -----------------------------

def load_onnx(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


yolo = load_onnx(YOLO_MODEL)
aesthetic = load_onnx(AESTHETIC_MODEL)


def detect_subjects(image):
    """Run YOLO to detect people, faces, sports objects."""
    img = cv2.resize(image, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]

    outputs = yolo.run(None, {"images": img})[0]
    score = float(np.mean(outputs[:, 4]))  # objectness
    return max(score, 0.0)


def aesthetic_score(image):
    """Run aesthetic model."""
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]

    score = aesthetic.run(None, {"input": img})[0][0][0]
    return float(score)


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

    # Normalize sharpness
    sharp_norm = min(sharp / 500.0, 1.0)

    if profile == "sports":
        final = (
            0.45 * sharp_norm +
            0.15 * expo +
            0.10 * noise +
            0.10 * blur +
            0.10 * subj +
            0.10 * aesth
        )
    else:  # burlesque / live music
        final = (
            0.25 * sharp_norm +
            0.20 * expo +
            0.10 * noise +
            0.05 * blur +
            0.20 * subj +
            0.20 * aesth
        )

    return final, sharp_norm, expo, noise, blur, subj, aesth


# -----------------------------
# METADATA WRITING
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
# MAIN PIPELINE
# -----------------------------

def process_directory(folder, profile):
    folder = Path(folder)
    results = []

    for path in folder.rglob("*"):
        if path.suffix.lower() not in SUPPORTED_RAW + SUPPORTED_IMG:
            continue

        print(f"Processing {path.name}...")

        img = extract_preview(path)
        if img is None:
            print("  Could not load preview.")
            continue

        score, sharp, expo, noise, blur, subj, aesth = score_image(img, profile)

        # Auto reject logic
        auto_reject = (
            sharp < 0.15 or
            expo < 0.2 or
            blur < 0.1
        )

        if auto_reject:
            set_xmp_flag(path, reject=1, label="Red")
            print("  → AUTO REJECT")
        else:
            results.append((path, score))

    # Sort and mark top 10%
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        top_n = max(1, len(results) // 10)

        for i, (path, score) in enumerate(results):
            if i < top_n:
                set_xmp_flag(path, pick=1, label="Green", rating=5)
                print(f"TOP PICK → {path.name}")
            else:
                set_xmp_flag(path, rating=3)

    print("Done.")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Image Culling Tool")
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument("--profile", choices=["sports", "burlesque"], default="sports")
    args = parser.parse_args()

    process_directory(args.folder, args.profile)


if __name__ == "__main__":
    main()
