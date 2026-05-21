from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from .scoring_profiles import combine_scores
from .utils.image_loader import is_supported, extract_preview
from .utils.metrics import (
    variance_of_laplacian,
    exposure_score,
    noise_score,
    blur_score,
)
from .utils.metadata import set_xmp_flag
from .utils.html_report import generate_html_report

YOLO_MODEL = str(Path(__file__).parent / "models" / "yolov8n.onnx")
AESTHETIC_MODEL = str(Path(__file__).parent / "models" / "laion_aesthetic.onnx")
FACE_MODEL = str(Path(__file__).parent / "models" / "face_detector.onnx")
HTML_REPORT_NAME = "ai_photo_cull_report.html"


def _load_onnx(model_path: str):
    providers = []
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


_yolo = None
_aesthetic = None
_face_det = None


def _get_yolo():
    global _yolo
    if _yolo is None:
        _yolo = _load_onnx(YOLO_MODEL)
    return _yolo


def _get_aesthetic():
    global _aesthetic
    if _aesthetic is None:
        _aesthetic = _load_onnx(AESTHETIC_MODEL)
    return _aesthetic


def _get_face_det():
    global _face_det
    if _face_det is None:
        _face_det = _load_onnx(FACE_MODEL)
    return _face_det


def detect_subjects(image):
    yolo = _get_yolo()
    img = cv2.resize(image, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]
    outputs = yolo.run(None, {yolo.get_inputs()[0].name: img})[0]
    score = float(np.mean(outputs[:, 4]))
    return max(score, 0.0)


def aesthetic_score(image):
    aesthetic = _get_aesthetic()
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]
    outputs = aesthetic.run(None, {aesthetic.get_inputs()[0].name: img})[0]
    return float(outputs[0][0])


def face_eye_sharpness(image):
    face_det = _get_face_det()
    h, w = image.shape[:2]
    img = cv2.resize(image, (320, 320))
    img_norm = img.astype(np.float32) / 255.0
    img_norm = np.transpose(img_norm, (2, 0, 1))[None, :, :, :]

    outputs = face_det.run(None, {face_det.get_inputs()[0].name: img_norm})[0]
    if outputs.shape[0] == 0:
        return 0.0

    best = outputs[np.argmax(outputs[:, 4])]
    x1, y1, x2, y2, conf = best
    if conf < 0.3:
        return 0.0

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


def score_image(image, profile):
    sharp = variance_of_laplacian(image)
    expo = exposure_score(image)
    noise = noise_score(image)
    blur = blur_score(image)
    subj = detect_subjects(image)
    aesth = aesthetic_score(image)
    face_sharp = face_eye_sharpness(image)

    sharp_norm = min(sharp / 500.0, 1.0)

    final = combine_scores(
        profile,
        sharp_norm,
        face_sharp,
        expo,
        noise,
        blur,
        subj,
        aesth,
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


def process_directory(folder, profile):
    folder = Path(folder)
    scored = []

    for path in folder.rglob("*"):
        if not is_supported(path):
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

    report_path = folder / HTML_REPORT_NAME
    generate_html_report(scored, report_path)
    print(f"HTML report written to: {report_path}")
