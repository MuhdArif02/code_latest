"""
=============================================================
Dashboard_FastAPI_MethodB.py
METHOD B  —  YOLO + pyzbar + CNN quality classifier
=============================================================
Comparison against Method A (Dashboard_FastAPI.py)

Pipeline:
  Camera → Center crop → YOLOv8 (locate QR) →
  pyzbar (decode QR data) → CNN (quality: PASS/REJECT) →
  Verdict smoother → FastAPI WebSocket dashboard

Install dependencies:
  pip install ultralytics torch torchvision
  pip install pyzbar opencv-python fastapi uvicorn

To train the CNN quality classifier:
  Run train_cnn_quality.py first (see bottom of this file)

Run:
  python Dashboard_FastAPI_MethodB.py
  Open http://localhost:8096 in browser
=============================================================
"""

import sys
import os
import time
import json
import base64
import threading
import asyncio
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from collections import deque
from pyzbar.pyzbar import decode, ZBarSymbol
from ultralytics import YOLO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import subprocess


# =========================================================
# CONFIG  (same as Method A where possible)
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

# YOLO config
YOLO_MODEL_PATH  = "yolov8n.pt"       # use yolov8n.engine on Jetson Nano
YOLO_CONF        = 0.30               # minimum detection confidence
YOLO_QR_CLASS    = 0                  # class index for QR code in your model
                                       # (0 if you trained a single-class QR detector)

# CNN quality classifier config
CNN_MODEL_PATH   = "qr_quality_cnn.pth"   # trained by train_cnn_quality.py
CNN_INPUT_SIZE   = 64                      # resize QR patch to 64x64 before CNN
CNN_PASS_THRESH  = 0.5                     # sigmoid output >= 0.5 → PASS

# Verdict smoother (same as Method A)
SMOOTH_WINDOW    = 5
PASS_MAJORITY    = 3
REJECT_MAJORITY  = 2

WS_HOST = "0.0.0.0"
WS_PORT = 8096                        # different port from Method A (8095)

COLOR = {
    "PASS":   (0, 220, 80),
    "REJECT": (0, 0, 255),
    "NO":     (0, 165, 255),
}

DASHBOARD_HTML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dashboard.html"
)

# =========================================================
# GLOBAL STATE
# =========================================================
latest_payload = {
    "timestamp":   time.time(),
    "status":      "NO",
    "error_type":  "system_start",
    "qr_data":     "",
    "confidence":  0.0,
    "bbox":        [],
    "metrics":     {},
    "yolo_conf":   0.0,
    "cnn_score":   0.0,
}

app = FastAPI(title="QR Inspection Dashboard — Method B")
connected_clients = set()

_reader    = None
_processor = None
_streamer  = None


# =========================================================
# CNN QUALITY CLASSIFIER  (MobileNetV2-style lightweight net)
# =========================================================
class QRQualityCNN(nn.Module):
    """
    Lightweight CNN that takes a 64x64 grayscale QR patch
    and outputs a single score (sigmoid):
      >= CNN_PASS_THRESH → PASS
      <  CNN_PASS_THRESH → REJECT
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_cnn_model(path):
    model = QRQualityCNN()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"CNN model loaded from {path}")
    else:
        print(f"WARNING: CNN model not found at {path}. Using untrained model.")
        print("Run train_cnn_quality.py first to train the model.")
    model.eval()
    return model


cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
    transforms.ToTensor(),
])


def cnn_predict(model, patch_bgr):
    """
    Run CNN on a BGR QR patch.
    Returns (verdict, score) where score is 0.0-1.0.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return "REJECT", 0.0

    try:
        tensor = cnn_transform(patch_bgr).unsqueeze(0)   # (1,1,64,64)
        with torch.no_grad():
            score = model(tensor).item()
        verdict = "PASS" if score >= CNN_PASS_THRESH else "REJECT"
        return verdict, round(score, 3)
    except Exception as e:
        print(f"CNN inference error: {e}")
        return "REJECT", 0.0


# =========================================================
# YOLO QR LOCATOR
# =========================================================
def load_yolo_model(path):
    if os.path.exists(path):
        model = YOLO(path)
        print(f"YOLO model loaded from {path}")
    else:
        # Download default YOLOv8n if not found
        print(f"YOLO model not found at {path}. Downloading yolov8n...")
        model = YOLO("yolov8n.pt")
        print("Note: yolov8n is a general detector. For best results,")
        print("train a dedicated QR detector on your dataset.")
    return model


def yolo_detect_qr(yolo_model, frame):
    """
    Run YOLO on frame. Returns list of (bbox, confidence) tuples.
    bbox = (x1, y1, x2, y2) in pixel coords.
    """
    results = yolo_model(frame, verbose=False, conf=YOLO_CONF)
    detections = []

    for r in results:
        for box in r.boxes:
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            detections.append(((x1, y1, x2, y2), conf))

    return detections


def crop_patch(frame, bbox, padding=10):
    """Safely crop a QR patch from frame with optional padding."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]


# =========================================================
# HELPERS  (same as Method A)
# =========================================================
def center_crop(frame, crop_w, crop_h):
    h_img, w_img = frame.shape[:2]
    crop_w = min(crop_w, w_img)
    crop_h = min(crop_h, h_img)
    x = (w_img - crop_w) // 2
    y = (h_img - crop_h) // 2
    return frame[y:y+crop_h, x:x+crop_w], x, y


def decode_qr_pyzbar(patch):
    """Try to decode QR from patch using pyzbar."""
    if patch is None or patch.size == 0:
        return None

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Try gray, Otsu, adaptive
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap    = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

    for img in (gray, otsu, adap):
        codes = decode(img, symbols=SYMBOLS)
        if codes:
            data = codes[0].data.decode("utf-8", errors="ignore").strip()
            if data:
                return data

    return None


def draw_result_yolo(frame, bbox, verdict, qr_data, yolo_conf, cnn_score):
    """Draw YOLO bounding box and verdict on frame."""
    color = COLOR[verdict]
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{verdict} | YOLO:{yolo_conf:.2f} CNN:{cnn_score:.2f}"
    cv2.putText(frame, label, (x1, max(20, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if qr_data:
        cv2.putText(frame, f"Data: {qr_data[:30]}", (x1, y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def make_payload_b(verdict, error_type, qr_data, bbox,
                   yolo_conf, cnn_score, prep_info):
    return {
        "timestamp":   time.time(),
        "status":      verdict,
        "error_type":  error_type,
        "qr_data":     qr_data,
        "confidence":  cnn_score,
        "bbox":        list(bbox) if bbox else [],
        "metrics":     {
            "yolo_conf": yolo_conf,
            "cnn_score": cnn_score,
        },
        "method":      "B",
    }


# =========================================================
# SMOOTHER  (same as Method A)
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
# QR COUNT LOOP  (same as Method A)
# =========================================================
class QRCountLoop:
    def __init__(self):
        self.waiting_new_qr = True
        self.pass_count     = 0
        self.reject_count   = 0
        self.total_count    = 0
        self.no_qr_counter  = 0
        self.NO_QR_THRESHOLD = 5

    def update(self, verdict):
        counted_now = False
        if verdict == "NO":
            self.no_qr_counter += 1
            if self.no_qr_counter >= self.NO_QR_THRESHOLD:
                self.waiting_new_qr = True
            return counted_now

        if verdict in ["PASS", "REJECT"]:
            self.no_qr_counter = 0
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
# USB CAMERA READER  (same as Method A)
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
# RTSP STREAMER  (same as Method A)
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
            self.rtsp_url
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("RTSPStreamer: started")
        except Exception as e:
            print(f"RTSPStreamer: failed to start — {e}")

    def write(self, frame):
        if self.proc is None:
            return
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


# =========================================================
# INSPECTION PROCESSOR  — METHOD B CORE
# =========================================================
class InspectionProcessorB(threading.Thread):
    """
    Method B pipeline per frame:
      1. Center crop
      2. YOLO → find QR bounding boxes
      3. For each detection:
           a. Crop QR patch
           b. pyzbar → decode QR data
           c. CNN → quality score (PASS/REJECT)
      4. Pick best verdict
      5. Smooth verdict
      6. Build payload → broadcast
    """
    def __init__(self, yolo_model, cnn_model):
        super().__init__(daemon=True)
        self.yolo_model  = yolo_model
        self.cnn_model   = cnn_model
        self.running     = False
        self.input_lock  = threading.Lock()
        self.latest_input = None
        self.latest_result = None
        self.result_lock = threading.Lock()
        self.smoother    = VerdictSmoother(
            window=SMOOTH_WINDOW,
            pass_thresh=PASS_MAJORITY,
            reject_thresh=REJECT_MAJORITY,
        )
        self.qr_counter     = QRCountLoop()
        self.processed_count = 0
        self.last_proc_time  = 0.0
        self.last_qr_data    = ""
        self.last_error_type = None
        self.last_cnn_score  = 0.0
        self.last_yolo_conf  = 0.0
        self.last_bbox       = None

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
                time.sleep(0.005)
                continue

            frame, frame_id = item
            t0 = time.perf_counter()

            # Step 1: center crop (same as Method A)
            crop_frame, _, _ = center_crop(frame, CROP_W, CROP_H)
            display_frame    = crop_frame.copy()

            # Step 2: YOLO detect QR locations
            detections = yolo_detect_qr(self.yolo_model, crop_frame)

            display_verdict    = "NO"
            display_bbox       = None
            display_error_type = None
            display_cnn_score  = 0.0
            display_yolo_conf  = 0.0
            qr_data            = ""

            if detections:
                # Process highest-confidence detection first
                detections.sort(key=lambda d: d[1], reverse=True)

                for bbox, yolo_conf in detections:
                    # Step 3a: crop QR patch
                    patch = crop_patch(crop_frame, bbox, padding=10)
                    if patch.size == 0:
                        continue

                    # Step 3b: pyzbar decode
                    decoded_data = decode_qr_pyzbar(patch)

                    # Step 3c: CNN quality check
                    cnn_verdict, cnn_score = cnn_predict(self.cnn_model, patch)

                    # Decide verdict for this detection
                    if decoded_data is None:
                        verdict    = "REJECT"
                        error_type = "unreadable_qr"
                    elif cnn_verdict == "REJECT":
                        verdict    = "REJECT"
                        error_type = "low_quality"
                    else:
                        verdict    = "PASS"
                        error_type = "none"

                    display_verdict    = verdict
                    display_bbox       = bbox
                    display_error_type = error_type
                    display_cnn_score  = cnn_score
                    display_yolo_conf  = yolo_conf
                    qr_data            = decoded_data or ""

                    # Draw on display frame
                    draw_result_yolo(
                        display_frame, bbox, verdict,
                        qr_data, yolo_conf, cnn_score
                    )

                    # Stop at first PASS, keep looking if REJECT
                    if verdict == "PASS":
                        break

            # Step 4: smooth verdict
            final_verdict = self.smoother.update(display_verdict)
            counted_now   = self.qr_counter.update(final_verdict)

            # Persist last known values (same pattern as Method A)
            if display_verdict != "NO":
                self.last_qr_data    = qr_data
                self.last_error_type = display_error_type
                self.last_cnn_score  = display_cnn_score
                self.last_yolo_conf  = display_yolo_conf
                self.last_bbox       = display_bbox

            eff_qr_data    = qr_data            if display_verdict != "NO" else self.last_qr_data
            eff_error_type = display_error_type  if display_verdict != "NO" else self.last_error_type
            eff_cnn_score  = display_cnn_score   if display_verdict != "NO" else self.last_cnn_score
            eff_yolo_conf  = display_yolo_conf   if display_verdict != "NO" else self.last_yolo_conf
            eff_bbox       = display_bbox         if display_verdict != "NO" else self.last_bbox

            # Step 5: build payload
            payload = make_payload_b(
                verdict    = final_verdict,
                error_type = eff_error_type,
                qr_data    = eff_qr_data,
                bbox       = eff_bbox,
                yolo_conf  = eff_yolo_conf,
                cnn_score  = eff_cnn_score,
                prep_info  = {},
            )

            payload["is_new_count"]  = counted_now
            payload["pass_count"]    = self.qr_counter.pass_count
            payload["reject_count"]  = self.qr_counter.reject_count
            payload["total_count"]   = self.qr_counter.total_count
            payload["no_count"]      = self.qr_counter.no_qr_counter

            _, buffer = cv2.imencode(
                '.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60]
            )
            payload["frame"] = base64.b64encode(buffer).decode('utf-8')

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
                    "qr_data":       qr_data,
                    "payload":       payload,
                    "proc_ms":       proc_ms,
                }


# =========================================================
# FASTAPI ROUTES  (same structure as Method A)
# =========================================================
@app.get("/")
async def dashboard():
    return FileResponse(DASHBOARD_HTML_PATH, media_type="text/html")


@app.get("/config.json")
async def config():
    return JSONResponse({"ws_port": WS_PORT, "method": "B"})


@app.post("/camera/start")
async def camera_start():
    global _reader, _processor, _streamer

    if _reader is not None:
        return {"status": "already_running"}

    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    cnn_model  = load_cnn_model(CNN_MODEL_PATH)

    _reader    = USBCameraReader(
        camera_index=USB_CAMERA_INDEX,
        width=USB_WIDTH, height=USB_HEIGHT, fps=USB_FPS
    )
    _processor = InspectionProcessorB(yolo_model, cnn_model)

    _reader.start()
    _reader.ready_event.wait(timeout=5.0)

    if not _reader.connected:
        _reader = _processor = None
        return {"status": "camera_failed"}

    _processor.start()

    _streamer = RTSPStreamer(
        rtsp_url=RTSP_PUBLISH_URL, width=CROP_W, height=CROP_H, fps=20
    )
    _streamer.start()

    threading.Thread(target=_feed_loop, daemon=True).start()
    return {"status": "started", "method": "B"}


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
        time.sleep(0.015)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print("WebSocket client connected")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
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
    print("  Method B — YOLO + pyzbar + CNN")
    print(f"  Dashboard: http://localhost:{WS_PORT}")
    print("=" * 55)
    uvicorn.run(app, host=WS_HOST, port=WS_PORT, ws="websockets-sansio")


# =========================================================
# CNN TRAINING SCRIPT (save as train_cnn_quality.py)
# =========================================================
"""
Save this section as train_cnn_quality.py and run it first.

Folder structure needed:
  qr_dataset/
    train/
      pass/    ← cropped QR patch images that are PASS
      reject/  ← cropped QR patch images that are REJECT
    val/
      pass/
      reject/

Usage:
  python train_cnn_quality.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CNN_INPUT_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
SAVE_PATH = "qr_quality_cnn.pth"

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder("qr_dataset/train", transform=transform)
val_ds   = datasets.ImageFolder("qr_dataset/val",   transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# class index: 0=pass, 1=reject (alphabetical)
# adjust CNN_PASS_THRESH accordingly

model     = QRQualityCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    for imgs, labels in train_loader:
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            labels = labels.float().unsqueeze(1)
            out    = model(imgs)
            preds  = (out >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS}  val_acc={acc:.1f}%")

torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
"""
