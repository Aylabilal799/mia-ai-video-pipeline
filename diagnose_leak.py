import sys
import subprocess
from PIL import Image

def ahash(path):
    img = Image.open(path).convert("L").resize((16, 16))
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return [1 if p > avg else 0 for p in pixels]

def extract_frame(video_path, timestamp, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
         "-frames:v", "1", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )

if __name__ == "__main__":
    video_path = sys.argv[1]
    timestamp = float(sys.argv[2])
    ref_path = "/root/deepseekyt/assets/mia_anchor.png"
    frame_path = "/tmp/diag_frame.jpg"

    extract_frame(video_path, timestamp, frame_path)
    h1 = ahash(frame_path)
    h2 = ahash(ref_path)
    hamming = sum(a != b for a, b in zip(h1, h2))
    print(f"Frame at {timestamp}s vs reference image:")
    print(f"Hamming distance = {hamming} / 256")
    print(f"Current threshold = 3.0  -->  {'WOULD match (<=3)' if hamming <= 3.0 else 'would NOT match (>3)'}")
