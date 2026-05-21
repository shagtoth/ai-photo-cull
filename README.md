📸 AI Photo Cull
Local, GPU‑accelerated AI culling for sports, derby, burlesque, and live‑gig photography.
ai-photo-cull is a fully local, privacy‑preserving image‑culling tool designed for high‑volume RAW workflows. 

It evaluates:

Sharpness
Face/Eye sharpness
Exposure & noise
Motion blur
Subject detection (YOLO)
Aesthetic appeal (LAION)
Composition cues

…and writes ON1‑compatible XMP metadata:

⭐ Ratings 
🎨 Color labels 
✔ Picks 
❌ Rejects 

It also generates a clean HTML contact sheet for fast review.

Runs entirely on your Mac — no cloud, no uploads.

🚀 Features
✔ Local AI scoring
Uses ONNX models accelerated via Apple Silicon (M‑series).

✔ Three tuned profiles
sports — action, clarity, subject isolation

burlesque — expression, lighting, aesthetic

derby — pack density, motion, face sharpness

✔ Auto‑reject
Culls obvious misfires (out‑of‑focus, underexposed, heavy blur).

✔ ON1 Photo RAW integration
Writes XMP metadata that ON1 automatically syncs into .on1 sidecars.

✔ HTML contact sheet
Quick visual overview of all scores and flags.

🛠 Installation
1. Install Python 3.11 (required)
Python 3.14 is not supported by onnxruntime-silicon or rawpy.

bash
brew install python@3.11

2. Run the installer script
This sets up everything:

Python 3.11 virtualenv

Dependencies

ONNX models
CLI command
Folder structure

bash
chmod +x install_ai_photo_cull.sh
./install_ai_photo_cull.sh

After installation, you can run:

bash
ai-photo-cull /path/to/shoot --profile derby

📂 Folder Structure
Code
ai-photo-cull/
│
├── ai_photo_cull/
│   ├── cli.py
│   ├── core.py
│   ├── scoring_profiles.py
│   ├── models/
│   │   ├── yolov8n.onnx
│   │   ├── laion_aesthetic.onnx
│   │   └── face_detector.onnx
│   └── utils/
│       ├── image_loader.py
│       ├── metrics.py
│       ├── metadata.py
│       └── html_report.py
│
├── install_ai_photo_cull.sh
├── pyproject.toml
└── README.md

🧪 Usage
Basic usage
bash
ai-photo-cull /path/to/images

With a profile
bash
ai-photo-cull /path/to/images --profile sports
ai-photo-cull /path/to/images --profile burlesque
ai-photo-cull /path/to/images --profile derby

Output

XMP metadata written directly into your images
ON1 Photo RAW picks up the changes automatically
HTML report saved as:

Code
/path/to/images/ai_photo_cull_report.html
Open it:

bash
open ai_photo_cull_report.html

🧠 Scoring Logic
Each image is evaluated on:

Metric	Description
Sharpness	Laplacian variance
Face/Eye Sharpness	SCRFD face detection + ROI sharpness
Exposure	Histogram clipping
Noise	Laplacian variance of blurred image
Motion Blur	FFT high‑frequency content
Subject Detection	YOLOv8n objectness
Aesthetic Score	LAION aesthetic predictor


Each profile applies different weights.

🎨 ON1 Integration
ON1 Photo RAW reads:

XMP:Rating
XMP:Label
XMP:Pick
XMP:Reject

This tool writes those fields directly.

ON1 automatically updates .on1 sidecars when it sees the XMP changes.

🧩 Troubleshooting

❗ “onnxruntime-silicon has no wheels for cp314”
You are using Python 3.14.

Fix: Install Python 3.11:

bash
brew install python@3.11
Then recreate your virtualenv.

❗ “rawpy failed to load RAW file”
Some RAW formats require updated libraw.
Try updating:

bash
brew upgrade rawpy
❗ “ai-photo-cull: command not found”
Re-run the installer or add this to your shell:

bash
echo "source ~/ai-photo-cull-env/bin/activate" >> ~/.zshrc

📜 License
MIT License.

❤️ Contributing
Pull requests welcome — especially:

New scoring profiles
New ONNX models
Better face/eye detection
Sport‑specific tuning (derby, polo, soccer, etc.)
GPU optimizations
