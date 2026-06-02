"""
=============================================================
Dashboard_FastAPI_MethodC.py
METHOD C  —  Zero-DCE + Retinex Hybrid Preprocessing
=============================================================
Based exactly on Dashboard_FastAPI.py (Method A).
Only change: auto_retinex_if_needed() now uses Zero-DCE
for very dark images (brightness < 50), then Retinex polishes.

Pipeline:
  Camera → Center crop → Right-half ROI →
  [Zero-DCE if very dark] → [Retinex if dark] →
  Otsu + Adaptive threshold →
  pyzbar decode → Quality checks →
  Verdict smoother → FastAPI WebSocket dashboard

Port: 8096  (Method A = 8095, run both for comparison)

Install:
  pip install torch torchvision
  Download zero_dce.pth (see instructions at bottom)

Run:
  python Dashboard_FastAPI_MethodC.py
  Open http://localhost:8096
=============================================================
"""

import sys
import os
import time
import json
import base64
import threading
import asyncio
import queue
import cv2
import numpy as np
import torch
import torch.nn as nn

from collections import deque
from pyzbar.pyzbar import decode, ZBarSymbol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import subprocess


# =========================================================
# CONFIG  (identical to Method A — only port changed)
# =========================================================
USB_CAMERA_INDEX = 0
USB_WIDTH        = 1280
USB_HEIGHT       = 720
USB_FPS          = 30

RTSP_PUBLISH_URL = "rtsp://127.0.0.1:8554/live"
FFMPEG_BIN       = r"C:\ffmpeg\bin\ffmpeg.exe"

SYMBOLS = [ZBarSymbol.QRCODE]

CROP_W = 960
CROP_H = 540

BLUR_THRESHOLD   = 55.0
ANGLE_PASS_MAX   = 35.0
MIN_DYNAMIC_RANGE = 35
MIN_SOLIDITY     = 0.75
MIN_SIDE         = 45
ZERODCE_PROCESS_SIZE = (320, 240)
SMOOTH_WINDOW    = 3
PASS_MAJORITY    = 2
REJECT_MAJORITY  = 2

WS_HOST = "0.0.0.0"
# WS_HOST = "192.168.100.157"
WS_PORT = 8096  
# WS_PORT = 1880        # ← different from Method A (8095)

COLOR = {
    "PASS":   (0, 220, 80),
    "REJECT": (0, 0, 255),
    "NO":     (0, 165, 255),
}

qr_detector = cv2.QRCodeDetector()

latest_payload = {
    "timestamp":   time.time(),
    "status":      "NO",
    "error_type":  "system_start",
    "qr_data":     "",
    "confidence":  0.0,
    "bbox":        [],
    "metrics":     {},
    "retinex_used": False,
    "zerodce_used": False,
}

DASHBOARD_HTML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dashboard.html"
)

app = FastAPI(title="QR Inspection Dashboard — Method C")
connected_clients = set()

_reader    = None
_processor = None
_streamer  = None


# =========================================================
# ZERO-DCE MODEL  ← NEW (only addition vs Method A)
# =========================================================
class ZeroDCE(nn.Module):
    """
    Zero-Reference Deep Curve Estimation (CVPR 2020)
    Lightweight 7-layer CNN that estimates pixel-wise
    enhancement curves for low-light image recovery.
    Input:  (1, 3, H, W) float32 RGB image [0,1]
    Output: (1, 3, H, W) enhanced image [0,1]
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        n = 32
        self.e_conv1 = nn.Conv2d(3,   n,   3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(n,   n,   3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(n,   n,   3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(n,   n,   3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(n*2, n,   3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(n*2, n,   3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(n*2, 24,  3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r = torch.split(x_r, 3, dim=1)   # 8 curve maps × 3 channels
        out = x
        for ri in r:
            out = out + ri * (out - out ** 2)   # iterative curve application
        return out


_zero_dce_model = None

def get_zero_dce():
    """Load Zero-DCE model once and cache it."""
    global _zero_dce_model
    if _zero_dce_model is None:
        _zero_dce_model = ZeroDCE()
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "zero_dce.pth"
        )
        if os.path.exists(weights_path):
            _zero_dce_model.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )
            print(f"Zero-DCE: pretrained weights loaded from {weights_path}")
        else:
            print(f"WARNING: zero_dce.pth not found at {weights_path}")
            print("Zero-DCE will run with random weights (poor quality).")
            print("Download: see instructions at bottom of this file.")
        _zero_dce_model.eval()
    return _zero_dce_model



# def apply_zero_dce(frame_bgr):
    """
    Run Zero-DCE on a BGR frame.
    Returns enhanced BGR frame (uint8).
    """
    # model = get_zero_dce()
    # # BGR → RGB, normalise to [0,1]
    # rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    # with torch.no_grad():
    #     enhanced = model(tensor)
    # out = enhanced.squeeze().permute(1, 2, 0).numpy()
    # out = np.clip(out * 255, 0, 255).astype(np.uint8)
    # return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
def apply_zero_dce(frame_bgr):
    model = get_zero_dce()
    h_orig, w_orig = frame_bgr.shape[:2]

    # Resize DOWN before CNN — 4x faster
    small = cv2.resize(frame_bgr, ZERODCE_PROCESS_SIZE)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    with torch.inference_mode():   # faster than no_grad
        enhanced = model(tensor)

    out = enhanced.squeeze().permute(1, 2, 0).numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # Resize back UP to original size
    return cv2.resize(out, (w_orig, h_orig))

# =========================================================
# RETINEX  (unchanged from Method A)
# =========================================================
def simple_retinex(img, sigma=25):
    img_f = img.astype(np.float32) + 1.0
    blur  = cv2.GaussianBlur(img_f, (0, 0), sigma)
    retinex = np.log(img_f) - np.log(blur + 1.0)
    out = np.zeros_like(retinex)
    for c in range(3):
        out[:, :, c] = cv2.normalize(
            retinex[:, :, c], None, 0, 255, cv2.NORM_MINMAX
        )
    return np.uint8(out)


# =========================================================
# HYBRID PREPROCESSING  ← KEY CHANGE vs Method A
# =========================================================
def auto_retinex_if_needed(frame):
    """
    Method C hybrid preprocessing:
      brightness < 50   → Zero-DCE (AI) + Retinex polish
      brightness < 110
        or contrast < 50 → Retinex only (same as Method A)
      otherwise          → no enhancement
    """
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    zerodce_used = False

    if brightness < 50:
        # Very dark → Zero-DCE recovers hidden detail first
        try:
            enhanced     = apply_zero_dce(frame)
            enhanced     = simple_retinex(enhanced, sigma=25)  # polish
            zerodce_used = True
            used_retinex = True
            print(f"[MethodC] Zero-DCE + Retinex applied (brightness={brightness:.1f})")
        except Exception as e:
            print(f"[MethodC] Zero-DCE failed ({e}), fallback to Retinex")
            enhanced     = simple_retinex(frame, sigma=25)
            used_retinex = True

    elif brightness < 110 or contrast < 50:
        # Moderately dark → Retinex only (identical to Method A)
        enhanced     = simple_retinex(frame, sigma=25)
        used_retinex = True

    else:
        # Good lighting → no enhancement
        enhanced     = frame.copy()
        used_retinex = False

    return enhanced, used_retinex, brightness, contrast, zerodce_used


def preprocess_variants(frame):
    """
    Returns enhanced frame + thresholded variants for pyzbar.
    Same as Method A except auto_retinex_if_needed now returns zerodce_used.
    """
    enhanced, used_retinex, brightness, contrast, zerodce_used = \
        auto_retinex_if_needed(frame)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adap = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    info = {
        "used_retinex": used_retinex,
        "zerodce_used": zerodce_used,
        "brightness":   brightness,
        "contrast":     contrast,
    }
    return enhanced, gray, otsu, adap, info


# =========================================================
# HELPERS  (unchanged from Method A)
# =========================================================
def measure_blur(gray_roi):
    if gray_roi.size == 0:
        return 0.0
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var()


def get_angle_from_pts(pts):
    rect = cv2.minAreaRect(pts.astype(np.float32))
    angle = rect[2]
    if rect[1][0] >= rect[1][1]:
        angle += 90
    if angle > 90:
        angle -= 180
    return angle


def normalize_qr_angle(angle):
    candidates = [0, 90, -90, 180, -180]
    closest = min(candidates, key=lambda x: abs(angle - x))
    return angle - closest


def get_safe_roi(img, x, y, w, h):
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    return img[y1:y2, x1:x2]


def polygon_solidity(pts):
    cnt      = pts.reshape(-1, 1, 2).astype(np.int32)
    area     = cv2.contourArea(cnt)
    hull     = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return area / hull_area


def center_crop(frame, crop_w, crop_h):
    h_img, w_img = frame.shape[:2]
    crop_w = min(crop_w, w_img)
    crop_h = min(crop_h, h_img)
    x = (w_img - crop_w) // 2
    y = (h_img - crop_h) // 2
    return frame[y:y + crop_h, x:x + crop_w], x, y


def decode_with_fallbacks(gray, otsu, adap):
    for img in (gray, otsu, adap):
        codes = decode(img, symbols=SYMBOLS)
        if codes:
            return codes
    return []


def detect_qr_shape(gray):
    ok, points = qr_detector.detect(gray)
    if not ok or points is None:
        return False, None
    return True, points[0].astype(np.float32)


# =========================================================
# QUALITY METRICS  (unchanged from Method A)
# =========================================================
def compute_quality_metrics(frame, pts):
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    enhanced, used_retinex, _, _, _ = auto_retinex_if_needed(frame)
    gray     = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray_roi = get_safe_roi(gray, x, y, w, h)
    bgr_roi  = get_safe_roi(enhanced, x, y, w, h)

    if gray_roi.size == 0 or bgr_roi.size == 0:
        return {
            "blur": 0.0, "dynamic": 0, "bimodal": 0.0,
            "solidity": 0.0, "angle": 0.0, "norm_angle": 0.0,
            "sat": 0.0, "w": w, "h": h, "retinex": used_retinex,
        }

    blur    = measure_blur(gray_roi)
    dynamic = int(gray_roi.max()) - int(gray_roi.min())
    total   = gray_roi.size
    extreme = int(np.sum(gray_roi < 64)) + int(np.sum(gray_roi > 192))
    bimodal = extreme / max(total, 1)

    hsv      = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    solidity  = polygon_solidity(pts)
    angle     = get_angle_from_pts(pts)
    norm_angle = normalize_qr_angle(angle)

    return {
        "blur": blur, "dynamic": dynamic, "bimodal": bimodal,
        "solidity": solidity, "angle": angle, "norm_angle": norm_angle,
        "sat": mean_sat, "w": w, "h": h, "retinex": used_retinex,
    }


def inspect_decoded_qr(frame, code):
    """Quality inspection — identical to Method A."""
    if len(code.polygon) != 4:
        return "REJECT", None, "irregular_polygon", {}, ""

    pts  = np.array([[p.x, p.y] for p in code.polygon], dtype=np.float32)
    data = code.data.decode("utf-8", errors="ignore").strip()
    metrics = compute_quality_metrics(frame, pts)

    if not data:
        return "REJECT", pts, "unreadable_data", metrics, ""
    if min(metrics["w"], metrics["h"]) < MIN_SIDE:
        return "REJECT", pts, "too_small", metrics, data
    if metrics["blur"] < BLUR_THRESHOLD:
        return "REJECT", pts, "blur", metrics, data
    if abs(metrics["norm_angle"]) > ANGLE_PASS_MAX:
        return "REJECT", pts, "rotation_error", metrics, data
    if not metrics["retinex"]:
        if metrics["dynamic"] < MIN_DYNAMIC_RANGE:
            return "REJECT", pts, "low_contrast", metrics, data
    else:
        if metrics["dynamic"] < 25:
            return "REJECT", pts, "very_low_contrast", metrics, data
    if metrics["solidity"] < MIN_SOLIDITY:
        return "REJECT", pts, "distorted", metrics, data

    return "PASS", pts, "none", metrics, data


def draw_result(frame, verdict, pts, reason, metrics):
    """Draw bounding box and verdict — unchanged from Method A."""
    color = COLOR[verdict]
    if pts is not None and len(pts) == 4:
        pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_i], True, color, 3)
        x = int(np.min(pts[:, 0]))
        y = int(np.min(pts[:, 1]))
        h = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        cv2.putText(
            frame, f"{verdict} | {reason}",
            (x, max(25, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )
        if metrics:
            line = (
                f"B:{metrics.get('blur',0):.0f}  "
                f"Dyn:{metrics.get('dynamic',0)}  "
                f"nA:{metrics.get('norm_angle',0):.1f}"
            )
            cv2.putText(
                frame, line,
                (x, y + h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )


def make_payload(verdict, error_type, qr_data, pts, metrics, prep_info):
    """Build WebSocket payload — adds zerodce_used flag."""
    bbox = []
    if pts is not None and len(pts) == 4:
        bbox = pts.astype(int).tolist()

    blur_score    = float(metrics.get("blur",    0.0))
    dynamic_score = float(metrics.get("dynamic", 0.0))
    confidence    = round(
        min(1.0, max(0.0,
            (blur_score / 120.0) * 0.5 + (dynamic_score / 100.0) * 0.5
        )), 3
    ) if verdict != "NO" and metrics else 0.0

    return {
        "timestamp":    time.time(),
        "status":       verdict,
        "error_type":   None if verdict == "NO" else error_type,
        "qr_data":      qr_data,
        "confidence":   confidence,
        "bbox":         bbox,
        "metrics":      {k: round(float(v), 3) if isinstance(v, float)
                         else v for k, v in metrics.items()},
        "retinex_used": prep_info.get("used_retinex", False),
        "zerodce_used": prep_info.get("zerodce_used", False),   # ← NEW
        "brightness":   round(prep_info.get("brightness", 0.0), 2),
        "contrast":     round(prep_info.get("contrast",   0.0), 2),
        "method":       "C",
    }


# =========================================================
# VERDICT SMOOTHER  (unchanged from Method A)
# =========================================================
class VerdictSmoother:
    def __init__(self, window=5, pass_thresh=3, reject_thresh=2):
        self.window        = deque(maxlen=window)
        self.pass_thresh   = pass_thresh
        self.reject_thresh = reject_thresh
        self.last          = "NO"

    def update(self, raw):
        self.window.append(raw)
        if self.window.count("REJECT") >= self.reject_thresh:
            self.last = "REJECT"
        elif self.window.count("PASS") >= self.pass_thresh:
            self.last = "PASS"
        elif self.window.count("NO") == len(self.window):
            self.last = "NO"
        return self.last


# =========================================================
# QR COUNT LOOP  (unchanged from Method A)
# =========================================================
class QRCountLoop:
    def __init__(self):
        self.waiting_new_qr  = True
        self.pass_count      = 0
        self.reject_count    = 0
        self.total_count     = 0
        self.no_qr_counter   = 0
        self.no_count        = 0
        self.NO_QR_THRESHOLD = 3

    def update(self, verdict):
        counted_now = False
        if verdict == "NO":
            self.no_qr_counter += 1
            if self.no_qr_counter >= self.NO_QR_THRESHOLD:
                self.no_count += 1
                self.waiting_new_qr = True
            return counted_now

        if verdict in ["PASS", "REJECT"]:
            self.no_qr_counter = 0
            self.no_count = 0
            if self.waiting_new_qr:
                if verdict == "PASS":
                    self.pass_count += 1
                elif verdict == "REJECT":
                    self.reject_count += 1
                self.total_count += 1
                counted_now = True
                self.waiting_new_qr = False
        return counted_now


# =========================================================
# USB CAMERA READER  (unchanged from Method A)
# =========================================================
class USBCameraReader(threading.Thread):
    def __init__(self, camera_index=0, width=1280, height=720, fps=30):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.width        = width
        self.height       = height
        self.fps          = fps
        self.cap          = None
        self.lock         = threading.Lock()
        self.latest_frame = None
        self.running      = False
        self.connected    = False
        self.frame_id     = 0
        self.ready_event  = threading.Event()

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not self.cap.isOpened():
            self.connected = False
            self.running   = False
            self.ready_event.set()
            print("USBCameraReader: failed to open camera")
            return

        self.connected = True
        self.ready_event.set()
        print("USBCameraReader: camera opened")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame
                self.frame_id    += 1

        if self.cap:
            self.cap.release()
        self.connected = False

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.frame_id
            return self.latest_frame.copy(), self.frame_id

    def stop(self):
        self.running = False


# =========================================================
# RTSP STREAMER  (unchanged from Method A)
# =========================================================
class RTSPStreamer:
    def __init__(self, rtsp_url, width, height, fps=20):
        self.rtsp_url = rtsp_url
        self.width    = width
        self.height   = height
        self.fps      = fps
        self.proc     = None
        self.lock     = threading.Lock()

    def start(self):
        cmd = [
            FFMPEG_BIN, "-re",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", "libx264",
            "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("RTSPStreamer: started")
        except Exception as e:
            print(f"RTSPStreamer: failed — {e}")

    def write(self, frame):
        if self.proc is None:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        try:
            with self.lock:
                self.proc.stdin.write(frame.tobytes())
        except Exception:
            pass

    def stop(self):
        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass
        self.proc = None
        print("RTSPStreamer: stopped")


# =========================================================
# INSPECTION PROCESSOR  (same structure as Method A)
# =========================================================
class InspectionProcessor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running      = False
        self.input_lock   = threading.Lock()
        self.latest_input = None
        self.latest_result = None
        self.result_lock  = threading.Lock()

        self.smoother = VerdictSmoother(
            window=SMOOTH_WINDOW,
            pass_thresh=PASS_MAJORITY,
            reject_thresh=REJECT_MAJORITY,
        )
        self.qr_counter      = QRCountLoop()
        self.processed_count = 0
        self.last_proc_time  = 0.0
        self.last_error_type = None
        self.last_metrics    = {}
        self.last_qr_data    = ""

    def submit_frame(self, frame, frame_id):
        with self.input_lock:
            self.latest_input = (frame, frame_id)

    def get_latest_result(self):
        with self.result_lock:
            return self.latest_result

    def stop(self):
        self.running = False

    def run(self):
        self.running = True

        while self.running:
            item = None
            with self.input_lock:
                if self.latest_input is not None:
                    item = self.latest_input
                    self.latest_input = None

            if item is None:
                time.sleep(0.001)
                continue

            frame, frame_id = item
            t0 = time.perf_counter()

            # Step 1: center crop (same as Method A)
            crop_frame, _, _ = center_crop(frame, CROP_W, CROP_H)
            h_crop, w_crop   = crop_frame.shape[:2]

            # Step 2: right-half ROI (same as Method A)
            right_frame = crop_frame[:, w_crop * 1 // 2:]

            # Step 3: hybrid preprocessing (Zero-DCE + Retinex)
            enhanced_frame, gray, otsu, adap, prep_info = \
                preprocess_variants(right_frame)

            # Step 4: display frame (same logic as Method A)
            if prep_info["brightness"] < 110 or prep_info["contrast"] < 50:
                display_frame = simple_retinex(crop_frame, sigma=12)
            else:
                display_frame = crop_frame.copy()

            # Draw ROI line
            cv2.line(
                display_frame,
                (w_crop * 1 // 2, 0), (w_crop * 1 // 2, h_crop),
                (0, 255, 255), 2
            )

            # Label left half — Entrance
            cv2.putText(
                display_frame, "Entrance",
                (w_crop // 4 - 60, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            cv2.arrowedLine(
            display_frame,
            (w_crop // 4 + 70, 28),   # start point (after text)
            (w_crop // 4 + 130, 28),  # end point (arrow tip)
            (0, 255, 255), 2, tipLength=0.4
            )
            # Label right half — Area of Detection
            cv2.putText(
                display_frame, "Area of Detection",
                (w_crop // 2 + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

            # # Show Zero-DCE indicator on display frame
            # if prep_info.get("zerodce_used"):
            #     cv2.putText(
            #         display_frame, "Zero-DCE ON",
            #         (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            #         0.6, (0, 200, 255), 2
            #     )

            display_verdict    = "NO"
            display_pts        = None
            display_error_type = None
            display_metrics    = {}
            qr_data            = ""

            # Step 5: QR decode (same as Method A)
            codes = decode_with_fallbacks(gray, otsu, adap)

            if codes:
                best = None
                for code in codes:
                    verdict, pts, error_type, metrics, data = \
                        inspect_decoded_qr(enhanced_frame, code)
                    if verdict == "REJECT":
                        best = (verdict, pts, error_type, metrics, data)
                        break
                    if best is None:
                        best = (verdict, pts, error_type, metrics, data)

                display_verdict, display_pts, display_error_type, \
                    display_metrics, qr_data = best
            else:
                display_verdict    = "NO"
                display_pts        = None
                display_error_type = None
                display_metrics    = {}

            # Step 6: smooth verdict
            final_verdict = self.smoother.update(display_verdict)
            counted_now   = self.qr_counter.update(final_verdict)

            # Persist last known good values
            if display_verdict != "NO":
                self.last_error_type = display_error_type
                self.last_metrics    = display_metrics
                self.last_qr_data    = qr_data

            eff_error_type = display_error_type if display_verdict != "NO" \
                else self.last_error_type
            eff_metrics    = display_metrics    if display_verdict != "NO" \
                else self.last_metrics
            eff_qr_data    = qr_data            if display_verdict != "NO" \
                else self.last_qr_data

            # Draw result on display frame
            if display_verdict != "NO" and display_pts is not None:
                offset_pts = display_pts.copy()
                offset_pts[:, 0] += w_crop * 1 // 2
                draw_result(
                    display_frame, display_verdict, offset_pts,
                    display_error_type, display_metrics
                )

            # Build payload
            payload = make_payload(
                verdict    = final_verdict,
                error_type = eff_error_type,
                qr_data    = eff_qr_data,
                pts        = display_pts,
                metrics    = eff_metrics,
                prep_info  = prep_info,
            )

            payload["is_new_count"]  = counted_now
            payload["pass_count"]    = self.qr_counter.pass_count
            payload["reject_count"]  = self.qr_counter.reject_count
            payload["total_count"]   = self.qr_counter.total_count
            payload["no_count"] = self.qr_counter.no_count

            # Encode frame
            _, buffer = cv2.imencode(
                '.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60]
            )
            payload["frame"] = base64.b64encode(buffer).decode('utf-8')

            # if counted_now:
            #     _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            #     payload["frame"] = base64.b64encode(buffer).decode('utf-8')
            # else:
            #     payload["frame"] = None


            send_payload_to_clients(payload)

            if _streamer is not None:
                _streamer.write(display_frame)

            proc_ms = (time.perf_counter() - t0) * 1000.0
            self.last_proc_time   = proc_ms
            self.processed_count += 1

            with self.result_lock:
                self.latest_result = {
                    "frame_id":      frame_id,
                    "display_frame": display_frame,
                    "final_verdict": final_verdict,
                    "display_error_type": display_error_type,
                    "qr_data":       qr_data,
                    "prep_info":     prep_info,
                    "payload":       payload,
                    "proc_ms":       proc_ms,
                    "processed_count": self.processed_count,
                }


# =========================================================
# FASTAPI ROUTES  (same as Method A)
# =========================================================
@app.get("/")
async def dashboard():
    return FileResponse(DASHBOARD_HTML_PATH, media_type="text/html")


@app.get("/config.json")
async def config():
    return JSONResponse({"ws_port": WS_PORT, "method": "C"})


@app.post("/camera/start")
async def camera_start():
    global _reader, _processor, _streamer

    if _reader is not None:
        return {"status": "already_running"}

    # Pre-load Zero-DCE at startup (avoids first-frame delay)
    print("Pre-loading Zero-DCE model...")
    get_zero_dce()

    _reader = USBCameraReader(
        camera_index=USB_CAMERA_INDEX,
        width=USB_WIDTH, height=USB_HEIGHT, fps=USB_FPS
    )
    _processor = InspectionProcessor()

    _reader.start()
    _reader.ready_event.wait(timeout=5.0)

    if not _reader.connected:
        _reader = _processor = None
        return {"status": "camera_failed"}

    _processor.start()

    _streamer = RTSPStreamer(
        rtsp_url=RTSP_PUBLISH_URL,
        width=CROP_W, height=CROP_H, fps=20
    )
    _streamer.start()

    threading.Thread(target=_feed_loop, daemon=True).start()
    return {"status": "started", "method": "C"}


@app.post("/camera/stop")
async def camera_stop():
    global _reader, _processor, _streamer

    if _reader:    _reader.stop();    _reader    = None
    if _processor: _processor.stop(); _processor = None
    if _streamer:  _streamer.stop();  _streamer  = None

    return {"status": "stopped"}


def _feed_loop():
    last_frame_id = -1
    while _reader is not None and _processor is not None:
        frame, frame_id = _reader.get_latest_frame()
        if frame is not None and frame_id != last_frame_id:
            last_frame_id = frame_id
            _processor.submit_frame(frame, frame_id)
        time.sleep(0.001)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print("WebSocket client connected")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    finally:
        connected_clients.discard(websocket)


async def broadcast_payload(payload):
    dead = []
    msg  = json.dumps(payload)
    for ws in connected_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.discard(ws)


def send_payload_to_clients(payload):
    global latest_payload
    latest_payload = payload
    if connected_clients:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(broadcast_payload(payload))
            else:
                loop.run_until_complete(broadcast_payload(payload))
        except RuntimeError:
            asyncio.run(broadcast_payload(payload))


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Method C — Zero-DCE + Retinex Hybrid")
    print(f"  Dashboard : http://localhost:{WS_PORT}")
    print(f"  Method A  : http://localhost:8095  (for comparison)")
    print("=" * 55)
    uvicorn.run(
        app, host=WS_HOST, port=WS_PORT,
        ws="websockets-sansio"
    )


# =============================================================
# HOW TO GET zero_dce.pth  (run once)
# =============================================================
# Option 1 — Download pretrained weights (recommended):
#
#   import urllib.request
#   url = "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth"
#   urllib.request.urlretrieve(url, "zero_dce.pth")
#   print("Downloaded zero_dce.pth")
#
# Option 2 — Install dependencies and run above:
#
#   pip install torch torchvision
#
# Place zero_dce.pth in the SAME folder as this file.
# =============================================================
