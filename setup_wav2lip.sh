#!/usr/bin/env bash
# Sets up Wav2Lip in its own isolated venv, invoked as a subprocess by the
# main pipeline. Does NOT touch the main project's Python environment.
#
# NOTE ON VERSIONS: Wav2Lip's own requirements.txt pins torch==1.1.0, which
# requires Python <=3.7 (EOL) and will fail to install on any modern system.
# This script installs current CPU-only torch/opencv instead -- Wav2Lip's
# model code (plain nn.Module, torch.cat/torch.load) is old but portable and
# runs fine on modern torch. This has NOT been run/tested on your actual
# machine -- run it yourself and check the output at the bottom.
set -euo pipefail

INSTALL_DIR="${1:-/root/deepseekyt/wav2lip}"
echo "Installing Wav2Lip (CPU) into: $INSTALL_DIR"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -d "Wav2Lip" ]; then
    git clone https://github.com/Rudrabha/Wav2Lip.git
fi
cd Wav2Lip

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
# CPU-only torch build -- much smaller download than the CUDA build, and
# correct since this server has no GPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python opencv-contrib-python numpy librosa==0.10.1 numba tqdm scipy

# --- Face detection weights (auto-downloaded by face_detection lib normally,
# but pre-fetch here so first real run doesn't stall/timeout) ---
mkdir -p face_detection/detection/sfd
if [ ! -f face_detection/detection/sfd/s3fd.pth ]; then
    echo ">>> Downloading face detector weights..."
    curl -L "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
        -o face_detection/detection/sfd/s3fd.pth
fi

mkdir -p checkpoints
echo ""
echo "=================================================================="
echo "MANUAL STEP REQUIRED — Wav2Lip's pretrained checkpoint is hosted on"
echo "Google Drive, not something I can fetch automatically. Download:"
echo ""
echo "  wav2lip_gan.pth (better visual quality, recommended):"
echo "  https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
echo "  (README links to the actual Drive/OneDrive files — links rotate,"
echo "   so grab the current one from there)"
echo ""
echo "Save it as:"
echo "  $INSTALL_DIR/Wav2Lip/checkpoints/wav2lip_gan.pth"
echo "=================================================================="
echo ""
echo "Once the checkpoint is in place, verify the install with:"
echo "  cd $INSTALL_DIR/Wav2Lip && source venv/bin/activate"
echo "  python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth \\"
echo "      --face <a test face image/video> --audio <a test wav/mp3>"
echo ""
echo "If that produces results/result_voice.mp4 with moving lips, you're set."
