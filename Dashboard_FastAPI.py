import sys
import time
import json
import base64
import threading
import asyncio
import queue
import cv2
import numpy as np

from collections import deque
from pyzbar.pyzbar import decode, ZBarSymbol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import subprocess


# =========================================================
# CONFIG
# =========================================================
USB_CAMERA_INDEX = 0
USB_WIDTH = 1280
USB_HEIGHT = 720
USB_FPS = 30

RTSP_PUBLISH_URL = "rtsp://127.0.0.1:8554/live"
FFMPEG_BIN = r"C:\ffmpeg\bin\ffmpeg.exe"

SYMBOLS = [ZBarSymbol.QRCODE]

CROP_W = 960
CROP_H = 540

BLUR_THRESHOLD = 55.0
ANGLE_PASS_MAX = 35.0
MIN_DYNAMIC_RANGE = 35
# MIN_BIMODAL_RATIO = 0.45
MIN_SOLIDITY = 0.75
MIN_SIDE = 45
# MAX_MEAN_SAT = 70.0

SMOOTH_WINDOW = 3
PASS_MAJORITY = 2
REJECT_MAJORITY = 2

WS_HOST = "0.0.0.0"
WS_PORT = 8095

COLOR = {
    "PASS": (0, 220, 80),
    "REJECT": (0, 0, 255),
    "NO": (0, 165, 255),
}

qr_detector = cv2.QRCodeDetector()

latest_payload = {
    "timestamp": time.time(),
    "status": "NO",
    "error_type": "system_start",
    "qr_data": "",
    "confidence": 0.0,
    "bbox": [],
    "metrics": {},
    "retinex_used": False
}

# =========================================================
# DASHBOARD HTML  →  served from dashboard.html (same directory)
# =========================================================
import os
DASHBOARD_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

# =========================================================
# WEBSOCKET SERVER (FastAPI)
# =========================================================
app = FastAPI(title="QR Inspection Dashboard")
connected_clients = set()

# Global handles for camera/processor/streamer
_reader = None
_processor = None
_streamer = None


@app.get("/")
async def dashboard():
    return FileResponse(DASHBOARD_HTML_PATH, media_type="text/html")


@app.get("/config.json")
async def config():
    """Lets dashboard.html discover the WS port without hardcoding."""
    return JSONResponse({"ws_port": WS_PORT})


@app.post("/camera/start")
async def camera_start():
    global _reader, _processor, _streamer

    if _reader is not None:
        return {"status": "already_running"}

    _reader = USBCameraReader(
        camera_index=USB_CAMERA_INDEX,
        width=USB_WIDTH,
        height=USB_HEIGHT,
        fps=USB_FPS
    )
    _processor = InspectionProcessor()

    _reader.start()
    _reader.ready_event.wait(timeout=5.0)

    if not _reader.connected:
        _reader = None
        _processor = None
        return {"status": "camera_failed"}

    _processor.start()

    _streamer = RTSPStreamer(
        rtsp_url=RTSP_PUBLISH_URL,
        width=CROP_W,
        height=CROP_H,
        fps=20
    )
    _streamer.start()

    # Start feeding frames from reader → processor
    threading.Thread(target=_feed_loop, daemon=True).start()

    return {"status": "started"}


@app.post("/camera/stop")
async def camera_stop():
    global _reader, _processor, _streamer

    if _reader:
        _reader.stop()
        _reader = None

    if _processor:
        _processor.stop()
        _processor = None

    if _streamer:
        _streamer.stop()
        _streamer = None

    return {"status": "stopped"}


def _feed_loop():
    """Continuously feeds newest camera frames into the processor."""
    last_frame_id = -1

    while _reader is not None and _processor is not None:
        frame, frame_id = _reader.get_latest_frame()

        if frame is not None and frame_id != last_frame_id:
            last_frame_id = frame_id
            _processor.submit_frame(frame, frame_id)

        time.sleep(0.001)  # ~66 fps poll


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
    dead_clients = []
    message = json.dumps(payload)

    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead_clients.append(ws)

    for ws in dead_clients:
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
# RETINEX / PREPROCESS  (unchanged)
# =========================================================
def simple_retinex(img, sigma=25):
    img_f = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
    retinex = np.log(img_f) - np.log(blur + 1.0)

    out = np.zeros_like(retinex)
    for c in range(3):
        out[:, :, c] = cv2.normalize(retinex[:, :, c], None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(out)


def auto_retinex_if_needed(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if brightness < 110 or contrast < 50:
        enhanced = simple_retinex(frame, sigma=25)
        used = True
    else:
        enhanced = frame.copy()
        used = False

    return enhanced, used, brightness, contrast


def preprocess_variants(frame):
    enhanced, used_retinex, brightness, contrast = auto_retinex_if_needed(frame)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    info = {"used_retinex": used_retinex, "brightness": brightness, "contrast": contrast}
    return enhanced, gray, otsu, adap, info


# =========================================================
# HELPERS  (unchanged)
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
    cnt = pts.reshape(-1, 1, 2).astype(np.int32)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
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
# METRICS / INSPECTION  (unchanged)
# =========================================================
# def compute_quality_metrics(frame, pts):
#     x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
#     enhanced, used_retinex, _, _ = auto_retinex_if_needed(frame)
#     gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
#     gray_roi = get_safe_roi(gray, x, y, w, h)
#     bgr_roi  = get_safe_roi(enhanced, x, y, w, h)

#     if gray_roi.size == 0 or bgr_roi.size == 0:
#         return {"blur": 0.0, "dynamic": 0, "bimodal": 0.0, "solidity": 0.0,
#                 "angle": 0.0, "norm_angle": 0.0, "sat": 0.0, "w": w, "h": h, "retinex": used_retinex}

#     blur    = measure_blur(gray_roi)
#     dynamic = int(gray_roi.max()) - int(gray_roi.min())
#     total   = gray_roi.size
#     extreme = int(np.sum(gray_roi < 64)) + int(np.sum(gray_roi > 192))
#     bimodal = extreme / max(total, 1)
#     hsv     = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
#     mean_sat= float(hsv[:, :, 1].mean())
#     solidity= polygon_solidity(pts)
#     angle   = get_angle_from_pts(pts)
#     norm_angle = normalize_qr_angle(angle)

#     return {"blur": blur, "dynamic": dynamic, "bimodal": bimodal, "solidity": solidity,
#             "angle": angle, "norm_angle": norm_angle, "sat": mean_sat,
#             "w": w, "h": h, "retinex": used_retinex}

def compute_quality_metrics(frame, pts):
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    enhanced, used_retinex, _, _ = auto_retinex_if_needed(frame)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray_roi = get_safe_roi(gray, x, y, w, h)

    if gray_roi.size == 0:
        return {
            "blur": 0.0,
            "dynamic": 0,
            "solidity": 0.0,
            "angle": 0.0,
            "norm_angle": 0.0,
            "w": w,
            "h": h,
            "retinex": used_retinex
        }

    blur = measure_blur(gray_roi)
    dynamic = int(gray_roi.max()) - int(gray_roi.min())
    solidity = polygon_solidity(pts)
    angle = get_angle_from_pts(pts)
    norm_angle = normalize_qr_angle(angle)

    return {
        "blur": blur,
        "dynamic": dynamic,
        "solidity": solidity,
        "angle": angle,
        "norm_angle": norm_angle,
        "w": w,
        "h": h,
        "retinex": used_retinex
    }

def inspect_decoded_qr(frame, code):
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
        # if metrics["bimodal"] < MIN_BIMODAL_RATIO:
        #     return "REJECT", pts, "contaminated", metrics, data
    else:
        if metrics["dynamic"] < 25:
            return "REJECT", pts, "very_low_contrast", metrics, data
        # if metrics["bimodal"] < 0.35:
        #     return "REJECT", pts, "heavy_contamination", metrics, data

    # if metrics["sat"] > MAX_MEAN_SAT:
    #     return "REJECT", pts, "colored_contamination", metrics, data
    if metrics["solidity"] < MIN_SOLIDITY:
        return "REJECT", pts, "distorted", metrics, data

    return "PASS", pts, "none", metrics, data


def draw_result(frame, verdict, pts, reason, metrics):
    color = COLOR[verdict]

    if pts is not None and len(pts) == 4:
        pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_i], True, color, 3)

        x = int(np.min(pts[:, 0]))
        y = int(np.min(pts[:, 1]))
        h = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))

        cv2.putText(frame, f"{verdict} | {reason}", (x, max(25, y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if metrics:
            line = (f"A:{metrics.get('angle', 0):.1f}  "
                    f"nA:{metrics.get('norm_angle', 0):.1f}  "
                    f"B:{metrics.get('blur', 0):.0f}  "
                    f"Dyn:{metrics.get('dynamic', 0)}")
            cv2.putText(frame, line, (x, y + h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def make_payload(verdict, error_type, qr_data, pts, metrics, prep_info):
    bbox = []
    if pts is not None and len(pts) == 4:
        pts_int = pts.astype(int)
        x = int(np.min(pts_int[:, 0]))
        y = int(np.min(pts_int[:, 1]))
        w = int(np.max(pts_int[:, 0]) - np.min(pts_int[:, 0]))
        h = int(np.max(pts_int[:, 1]) - np.min(pts_int[:, 1]))
        bbox = [x, y, w, h]

    confidence = 0.0
    if metrics:
        blur_score    = float(metrics.get("blur", 0.0))
        dynamic_score = float(metrics.get("dynamic", 0.0))
        confidence    = min(1.0, max(0.0, (blur_score / 120.0) * 0.5 + (dynamic_score / 100.0) * 0.5))

    return {
        "timestamp":    time.time(),
        "status":       verdict,
        "error_type":   error_type,
        "qr_data":      qr_data,
        "confidence":   round(confidence, 3),
        "bbox":         bbox,
        "metrics":      metrics,
        "retinex_used": prep_info.get("used_retinex", False),
        "brightness":   round(prep_info.get("brightness", 0.0), 2),
        "contrast":     round(prep_info.get("contrast", 0.0), 2),
    }

    # AFTER ✅
    # confidence = None
    # if verdict != "NO" and metrics:
    #     blur_score    = float(metrics.get("blur", 0.0))
    #     dynamic_score = float(metrics.get("dynamic", 0.0))
    #     confidence    = round(min(1.0, max(0.0, (blur_score / 120.0) * 0.5 + (dynamic_score / 100.0) * 0.5)), 3)

    # return {
    #     "timestamp":    time.time(),
    #     "status":       verdict,
    #     "error_type":   None if verdict == "NO" else error_type,
    #     "qr_data":      qr_data,
    #     "confidence":   confidence,
    #     "bbox":         bbox,
    #     "metrics":      metrics,
    #     "retinex_used": prep_info.get("used_retinex", False),
    #     "brightness":   round(prep_info.get("brightness", 0.0), 2),
    #     "contrast":     round(prep_info.get("contrast", 0.0), 2),
    # }


# =========================================================
# SMOOTHER  (unchanged)
# =========================================================
class VerdictSmoother:
    def __init__(self, window=5, pass_thresh=3, reject_thresh=2):
        self.window       = deque(maxlen=window)
        self.pass_thresh  = pass_thresh
        self.reject_thresh= reject_thresh
        self.last         = "NO"

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
# USB CAMERA READER  (unchanged)
# =========================================================
class USBCameraReader(threading.Thread):
    def __init__(self, camera_index=0, width=1280, height=720, fps=30):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.width  = width
        self.height = height
        self.fps    = fps
        self.cap    = None
        self.lock   = threading.Lock()
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
            print("USBCameraReader: failed to open USB camera")
            return

        self.connected = True
        self.ready_event.set()
        print("USBCameraReader: USB camera opened")

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
            self.cap = None

        self.connected = False
        print("USBCameraReader: stopped")

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.frame_id
            return self.latest_frame.copy(), self.frame_id

    def stop(self):
        self.running = False


# =========================================================
# RTSP STREAMER  (unchanged)
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
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"RTSPStreamer: publishing to {self.rtsp_url}")
        except FileNotFoundError:
            print("RTSPStreamer error: ffmpeg not found — check FFMPEG_BIN")
            self.proc = None

    def write(self, frame):
        if self.proc is None or self.proc.stdin is None or frame is None:
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
# QR COUNT LOOP (ONE QR = ONE COUNT)
# =========================================================
# class QRCountLoop:
#     def __init__(self):
#         self.waiting_new_qr = True
#         self.pass_count = 0
#         self.reject_count = 0
#         self.total_count = 0

#     def update(self, verdict):
#         counted_now = False

#         # Reset when NO QR
#         if verdict == "NO":
#             self.waiting_new_qr = True
#             return counted_now

#         # Count only FIRST detection
#         if self.waiting_new_qr:
#             if verdict == "PASS":
#                 self.pass_count += 1
#                 self.total_count += 1
#                 counted_now = True

#             elif verdict == "REJECT":
#                 self.reject_count += 1
#                 self.total_count += 1
#                 counted_now = True

#             # Lock until QR disappears
#             self.waiting_new_qr = False

#         return counted_now

# class QRCountLoop:
#     def __init__(self):
#         self.waiting_new_qr = True

#         self.pass_count = 0
#         self.reject_count = 0
#         self.total_count = 0

#         # 🔥 key fix
#         self.no_qr_counter = 0
#         self.NO_QR_THRESHOLD = 10   # adjust (8~15 recommended)

#     def update(self, verdict):
#         counted_now = False

#         # ===============================
#         # NO QR (may be noise)
#         # ===============================
#         if verdict == "NO":
#             self.no_qr_counter += 1

#             # only reset if QR gone for long enough
#             if self.no_qr_counter >= self.NO_QR_THRESHOLD:
#                 self.waiting_new_qr = True

#             return counted_now

#         # ===============================
#         # QR DETECTED
#         # ===============================
#         else:
#             self.no_qr_counter = 0  # reset noise counter

#             if self.waiting_new_qr:
#                 if verdict == "PASS":
#                     self.pass_count += 1
#                     self.total_count += 1
#                     counted_now = True

#                 elif verdict == "REJECT":
#                     self.reject_count += 1
#                     self.total_count += 1
#                     counted_now = True

#                 # lock until QR disappears stably
#                 self.waiting_new_qr = False

#         return counted_now

# =========================================================
# QR COUNT LOOP (STABLE ONE QR = ONE COUNT)
# =========================================================
class QRCountLoop:
    def __init__(self):
        self.waiting_new_qr = True

        self.pass_count = 0
        self.reject_count = 0
        self.total_count = 0

        self.no_qr_counter = 0
        self.NO_QR_THRESHOLD = 3

    def update(self, verdict):
        counted_now = False

        # NO QR must appear several frames before reset
        if verdict == "NO":
            self.no_qr_counter += 1

            if self.no_qr_counter >= self.NO_QR_THRESHOLD:
                self.waiting_new_qr = True

            return counted_now

        # QR detected
        if verdict in ["PASS", "REJECT"]:
            self.no_qr_counter = 0

            if self.waiting_new_qr:
                if verdict == "PASS":
                    self.pass_count += 1

                elif verdict == "REJECT":
                    self.reject_count += 1

                self.total_count += 1
                counted_now = True

                # lock until QR becomes NO for stable frames
                self.waiting_new_qr = False

        return counted_now

# =========================================================
# INSPECTION PROCESSOR  (unchanged)
# =========================================================
class InspectionProcessor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running      = False
        self.input_lock   = threading.Lock()
        self.latest_input = None
        self.latest_result= None
        self.result_lock  = threading.Lock()
        self.smoother     = VerdictSmoother(
            window=SMOOTH_WINDOW,
            pass_thresh=PASS_MAJORITY,
            reject_thresh=REJECT_MAJORITY,
        )
        self.qr_counter = QRCountLoop()
        self.processed_count = 0
        self.last_proc_time  = 0.0
        self.last_error_type  = None   # ← ADD
        self.last_metrics     = {}     # ← ADD
        self.last_qr_data     = ""     # ← ADD


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
        last_sent_verdict = None

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

            crop_frame, _, _ = center_crop(frame, CROP_W, CROP_H)
            h_crop, w_crop = crop_frame.shape[:2]
            # right_frame = crop_frame[:, w_crop*2//3:] #Right 1/3 only detect
            right_frame = crop_frame[:, w_crop*1//2:]

            # enhanced_frame, gray, otsu, adap, prep_info = preprocess_variants(crop_frame)
            enhanced_frame, gray, otsu, adap, prep_info = preprocess_variants(right_frame)
            # display_frame = enhanced_frame.copy()
            display_frame = crop_frame.copy()
            # cv2.line(display_frame, (w_crop*2//3, 0), (w_crop*2//3, h_crop), (0, 255, 255), 2)
            cv2.line(display_frame, (w_crop*1//2, 0), (w_crop*1//2, h_crop), (0, 255, 255), 2)

            display_verdict    = "NO"
            display_pts        = None
            display_error_type = None
            display_metrics    = {}
            qr_data            = ""

            codes = decode_with_fallbacks(gray, otsu, adap)

            if codes:
                best = None
                for code in codes:
                    verdict, pts, error_type, metrics, data = inspect_decoded_qr(enhanced_frame, code)
                    if verdict == "REJECT":
                        best = (verdict, pts, error_type, metrics, data)
                        break
                    if best is None:
                        best = (verdict, pts, error_type, metrics, data)
                display_verdict, display_pts, display_error_type, display_metrics, qr_data = best
            else:
                # ok, pts = detect_qr_shape(gray)
                # # if ok:
                # if ok and pts is not None and cv2.contourArea(pts.astype(np.int32)) > 10000:    
                #     display_verdict    = "REJECT"
                #     display_pts        = pts
                #     display_error_type = "unreadable_qr"
                #     display_metrics    = compute_quality_metrics(enhanced_frame, pts)
                # else:
                #     display_verdict    = "NO"
                #     display_error_type = "no_qr"
                
                display_verdict    = "NO"
                display_pts        = None
                display_error_type = None
                display_metrics    = {}

            final_verdict = self.smoother.update(display_verdict)
            counted_now = self.qr_counter.update(final_verdict)
            # Save last known good values when QR detected
            if display_verdict != "NO":
                self.last_error_type = display_error_type
                self.last_metrics    = display_metrics
                self.last_qr_data    = qr_data

            # Use saved values when current frame is NO
            eff_error_type = display_error_type if display_verdict != "NO" else self.last_error_type
            eff_metrics    = display_metrics    if display_verdict != "NO" else self.last_metrics
            eff_qr_data    = qr_data            if display_verdict != "NO" else self.last_qr_data

            # if display_verdict != "NO":
            #     draw_result(display_frame, display_verdict, display_pts,
            #                 display_error_type, display_metrics)
            if display_verdict != "NO" and display_pts is not None:
                # Offset pts by half width to correct position on full frame
                offset_pts = display_pts.copy()
                # offset_pts[:, 0] += w_crop *2// 3   # ← shift x coords to right half
                offset_pts[:, 0] += w_crop *1// 2 
                draw_result(display_frame, display_verdict, offset_pts,
                            display_error_type, display_metrics)               

            # payload = make_payload(
            #     verdict=final_verdict,
            #     error_type=display_error_type,
            #     qr_data=qr_data,
            #     pts=display_pts,
            #     metrics=display_metrics,
            #     prep_info=prep_info

            # payload = make_payload(
            #     verdict=final_verdict,
            #     error_type=display_error_type if display_verdict != "NO" else None,
            #     qr_data=qr_data,
            #     pts=display_pts,
            #     metrics=display_metrics if display_verdict != "NO" else {},
            #     prep_info=prep_info
            payload = make_payload(
                verdict=final_verdict,
                error_type=eff_error_type,
                qr_data=eff_qr_data,
                pts=display_pts,
                metrics=eff_metrics,
                prep_info=prep_info
            )

            # ✅ ADD HERE (DO NOT REMOVE ABOVE)
            # payload["counted_now"] = counted_now
            payload["is_new_count"] = counted_now
            payload["pass_count"] = self.qr_counter.pass_count
            payload["reject_count"] = self.qr_counter.reject_count
            payload["total_count"] = self.qr_counter.total_count

            payload["no_count"] = self.qr_counter.no_qr_counter

            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            payload["frame"] = base64.b64encode(buffer).decode('utf-8')
            
            # Only attach frame if verdict changed or every 3rd result
            if final_verdict != last_sent_verdict:
                last_sent_verdict = final_verdict
                payload["status"] = final_verdict

            send_payload_to_clients(payload)

            # Also push to RTSP streamer if running
            if _streamer is not None:
                _streamer.write(display_frame)

            proc_ms = (time.perf_counter() - t0) * 1000.0
            self.last_proc_time  = proc_ms
            self.processed_count += 1

            with self.result_lock:
                self.latest_result = {
                    "frame_id":          frame_id,
                    "display_frame":     display_frame,
                    "final_verdict":     final_verdict,
                    "display_error_type":display_error_type,
                    "qr_data":           qr_data,
                    "prep_info":         prep_info,
                    "payload":           payload,
                    "proc_ms":           proc_ms,
                    "processed_count":   self.processed_count,
                }


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    print(f"Starting QR Inspection Dashboard on http://0.0.0.0:{WS_PORT}")
    print(f"Open http://localhost:{WS_PORT} in your browser")
    uvicorn.run(app, host=WS_HOST, port=WS_PORT, ws="websockets-sansio")