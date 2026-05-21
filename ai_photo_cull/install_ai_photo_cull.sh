#!/bin/bash

echo "=== AI-PHOTO-CULL INSTALLER ==="

# 1. Install system dependencies
echo "Installing Homebrew dependencies..."
brew install python@3.11 exiftool wget unzip

# 2. Create virtual environment using Python 3.11
PY311=$(brew --prefix python@3.11)/bin/python3.11

echo "Creating Python 3.11 virtual environment..."
$PY311 -m venv ~/ai-photo-cull-env
source ~/ai-photo-cull-env/bin/activate

# 3. Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install rawpy opencv-python pillow numpy onnxruntime-silicon

# 4. Create folder structure
echo "Setting up ai-photo-cull folder..."
mkdir -p ~/ai-photo-cull/models

# 5. Download models
echo "Downloading YOLOv8n..."
wget -O ~/ai-photo-cull/models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

echo "Downloading LAION aesthetic model..."
wget -O ~/ai-photo-cull/models/laion_aesthetic.onnx \
  https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/onnx/aesthetic.onnx

echo "Downloading SCRFD face detector..."
wget -O /tmp/scrfd.zip \
  https://github.com/deepinsight/insightface/releases/download/onnx/models.zip
unzip -o /tmp/scrfd.zip -d /tmp/scrfd
cp /tmp/scrfd/scrfd_2.5g_bnkps.onnx ~/ai-photo-cull/models/face_detector.onnx

# 6. Install your CLI script
echo "Copying ai-photo-cull script..."
cp ai_cull.py ~/ai-photo-cull/

# 7. Create global symlink
echo "Creating global ai-photo-cull command..."
chmod +x ~/ai-photo-cull/ai_cull.py
sudo ln -sf ~/ai-photo-cull/ai_cull.py /usr/local/bin/ai-photo-cull

echo "=== INSTALL COMPLETE ==="
echo "Run with: ai-photo-cull /path/to/shoot --profile derby"
