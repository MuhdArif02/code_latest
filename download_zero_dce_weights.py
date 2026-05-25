"""
download_zero_dce_weights.py
Run this ONCE before starting Dashboard_FastAPI_MethodC.py
"""
import urllib.request
import os

SAVE_PATH = "zero_dce.pth"
URL = "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth"

if os.path.exists(SAVE_PATH):
    print(f"zero_dce.pth already exists — no download needed.")
else:
    print(f"Downloading Zero-DCE pretrained weights...")
    try:
        urllib.request.urlretrieve(URL, SAVE_PATH)
        print(f"Downloaded successfully → {SAVE_PATH}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Try manually downloading from:")
        print(URL)
        print(f"And rename the file to: {SAVE_PATH}")

print("Done. Place zero_dce.pth in the same folder as Dashboard_FastAPI_MethodC.py")
