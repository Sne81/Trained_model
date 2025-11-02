"""
Encode an arbitrary binary file (oral_cancer_model.h5) into a lossless PNG image.
- Prepends an 8-byte little-endian unsigned integer with the original file length.
- Packs bytes into a grayscale image (mode 'L') with a fixed width (1024 by default).
- Saves `oral_cancer_model.png` in the working directory.

Usage:
    python model_to_image.py

Reverse: run `extract_model_from_image.py` to recover the original .h5
"""
import os
import math
from pathlib import Path

IN_FILE = Path("oral_cancer_model.h5")
OUT_FILE = Path("oral_cancer_model.png")
WIDTH = 1024

try:
    from PIL import Image
except Exception:
    raise SystemExit("Pillow is required. Install with: pip install pillow")

if not IN_FILE.exists():
    raise SystemExit(f"Input file not found: {IN_FILE}")

data = IN_FILE.read_bytes()
orig_len = len(data)
print(f"Original model size: {orig_len} bytes")

# 8-byte header with original length
header = orig_len.to_bytes(8, byteorder='little')
buf = header + data

# compute dimensions
total_bytes = len(buf)
height = math.ceil(total_bytes / WIDTH)
pad = height * WIDTH - total_bytes
print(f"Encoding into image width={WIDTH}, height={height}, pad={pad} bytes")

# create bytearray and pad
arr = bytearray(buf)
if pad:
    arr.extend(b"\x00" * pad)

# convert to 2D array
import numpy as np
img = np.frombuffer(arr, dtype=np.uint8).reshape((height, WIDTH))
img_pil = Image.fromarray(img, mode='L')
img_pil.save(OUT_FILE)
print(f"Saved image: {OUT_FILE} ({OUT_FILE.stat().st_size} bytes)")
