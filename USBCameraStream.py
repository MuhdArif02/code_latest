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
from PyQt5 import uic
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyzbar.pyzbar import decode, ZBarSymbol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import subprocess



# =========================================================
# CONFIG
# =========================================================
USB_CAMERA_INDEX = 0           # usually 0, change to 1 if needed
USB_WIDTH = 1280
USB_HEIGHT = 720
USB_FPS = 30

RTSP_PUBLISH_URL = "rtsp://127.0.0.1:8554/live"
# RTSP_PUBLISH_URL = "rtsp://127.0.0.1:8554/live"
# FFMPEG_BIN = "ffmpeg"          # or full path: r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_BIN = r"C:\ffmpeg\bin\ffmpeg.exe"

SYMBOLS = [ZBarSymbol.QRCODE]

CROP_W = 960
CROP_H = 540

BLUR_THRESHOLD = 55.0
ANGLE_PASS_MAX = 35.0
MIN_DYNAMIC_RANGE = 35
MIN_BIMODAL_RATIO = 0.45
MIN_SOLIDITY = 0.75
MIN_SIDE = 45
MAX_MEAN_SAT = 70.0

SMOOTH_WINDOW = 5
PASS_MAJORITY = 3
REJECT_MAJORITY = 2

WS_HOST = "0.0.0.0"
WS_PORT = 8092

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
# WEBSOCKET SERVER
# =========================================================
app = FastAPI()
connected_clients = set()


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


def start_websocket_server():
    uvicorn.run(app, host=WS_HOST, port=WS_PORT, ws="websockets-sansio")

# _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
# payload["frame"] = base64.b64encode(buffer).decode('utf-8')

# send_payload_to_clients(payload)

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
# RETINEX / PREPROCESS
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

    info = {
        "used_retinex": used_retinex,
        "brightness": brightness,
        "contrast": contrast
    }

    return enhanced, gray, otsu, adap, info


# =========================================================
# HELPERS
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
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
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

    cropped = frame[y:y + crop_h, x:x + crop_w]
    return cropped, x, y


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

    pts = points[0].astype(np.float32)
    return True, pts


# =========================================================
# METRICS / INSPECTION
# =========================================================
def compute_quality_metrics(frame, pts):
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))

    enhanced, used_retinex, _, _ = auto_retinex_if_needed(frame)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray_roi = get_safe_roi(gray, x, y, w, h)
    bgr_roi = get_safe_roi(enhanced, x, y, w, h)

    if gray_roi.size == 0 or bgr_roi.size == 0:
        return {
            "blur": 0.0,
            "dynamic": 0,
            "bimodal": 0.0,
            "solidity": 0.0,
            "angle": 0.0,
            "norm_angle": 0.0,
            "sat": 0.0,
            "w": w,
            "h": h,
            "retinex": used_retinex,
        }

    blur = measure_blur(gray_roi)
    dynamic = int(gray_roi.max()) - int(gray_roi.min())

    total = gray_roi.size
    extreme = int(np.sum(gray_roi < 64)) + int(np.sum(gray_roi > 192))
    bimodal = extreme / max(total, 1)

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())

    solidity = polygon_solidity(pts)
    angle = get_angle_from_pts(pts)
    norm_angle = normalize_qr_angle(angle)

    return {
        "blur": blur,
        "dynamic": dynamic,
        "bimodal": bimodal,
        "solidity": solidity,
        "angle": angle,
        "norm_angle": norm_angle,
        "sat": mean_sat,
        "w": w,
        "h": h,
        "retinex": used_retinex,
    }


def inspect_decoded_qr(frame, code):
    if len(code.polygon) != 4:
        return "REJECT", None, "irregular_polygon", {}, ""

    pts = np.array([[p.x, p.y] for p in code.polygon], dtype=np.float32)
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

        if metrics["bimodal"] < MIN_BIMODAL_RATIO:
            return "REJECT", pts, "contaminated", metrics, data
    else:
        if metrics["dynamic"] < 25:
            return "REJECT", pts, "very_low_contrast", metrics, data

        if metrics["bimodal"] < 0.35:
            return "REJECT", pts, "heavy_contamination", metrics, data

    if metrics["sat"] > MAX_MEAN_SAT:
        return "REJECT", pts, "colored_contamination", metrics, data

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

        cv2.putText(
            frame,
            f"{verdict} | {reason}",
            (x, max(25, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        if metrics:
            line = (
                f"A:{metrics.get('angle', 0):.1f}  "
                f"nA:{metrics.get('norm_angle', 0):.1f}  "
                f"B:{metrics.get('blur', 0):.0f}  "
                f"Dyn:{metrics.get('dynamic', 0)}"
            )
            cv2.putText(
                frame,
                line,
                (x, y + h + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )


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
        blur_score = float(metrics.get("blur", 0.0))
        dynamic_score = float(metrics.get("dynamic", 0.0))
        confidence = min(
            1.0,
            max(0.0, (blur_score / 120.0) * 0.5 + (dynamic_score / 100.0) * 0.5)
        )

    payload = {
        "timestamp": time.time(),
        "status": verdict,
        "error_type": error_type,
        "qr_data": qr_data,
        "confidence": round(confidence, 3),
        "bbox": bbox,
        "metrics": metrics,
        "retinex_used": prep_info.get("used_retinex", False),
        "brightness": round(prep_info.get("brightness", 0.0), 2),
        "contrast": round(prep_info.get("contrast", 0.0), 2),
    }
    return payload


# =========================================================
# SMOOTHER
# =========================================================
class VerdictSmoother:
    def __init__(self, window=5, pass_thresh=3, reject_thresh=2):
        self.window = deque(maxlen=window)
        self.pass_thresh = pass_thresh
        self.reject_thresh = reject_thresh
        self.last = "NO"

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
# THREADED USB CAMERA READER
# Keeps only latest frame, older frames are overwritten
# =========================================================
class USBCameraReader(threading.Thread):
    def __init__(self, camera_index=0, width=1280, height=720, fps=30):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.connected = False
        self.frame_id = 0
        self.ready_event = threading.Event()

    def run(self):
        self.running = True

        # Windows: CAP_DSHOW often reduces delay for USB camera
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        # self.cap = cv2.VideoCapture(self.camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.connected = False
            self.running = False
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
                self.frame_id += 1

        if self.cap is not None:
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


# # =========================================================
# # RTSP STREAMER
# # Publishes processed frames to RTSP using FFmpeg
# # Requires an RTSP server such as MediaMTX running on rtsp://127.0.0.1:8554/live
# # =========================================================
# class RTSPStreamer:
#     def __init__(self, rtsp_url, width, height, fps=20):
                
#         self.streamer = None
#         self.stream_width = CROP_W
#         self.stream_height = CROP_H

#         # ❌ removed self.rtsp_url

#         self.reader = None
#         self.processor = None
#         # self.width = width
#         # self.height = height
#         # self.fps = fps
#         # self.proc = None
#         # self.lock = threading.Lock()

#     def start(self):
#         cmd = [
#             FFMPEG_BIN,
#             "-re",
#             "-f", "rawvideo",
#             "-pix_fmt", "bgr24",
#             "-s", f"{self.width}x{self.height}",
#             "-r", str(self.fps),
#             "-i", "-",
#             "-an",
#             "-c:v", "libx264",
#             "-preset", "ultrafast",
#             "-tune", "zerolatency",
#             "-pix_fmt", "yuv420p",
#             "-f", "rtsp",
#             "-rtsp_transport", "tcp",
#             self.rtsp_url
#         ]

#         self.proc = subprocess.Popen(
#             cmd,
#             stdin=subprocess.PIPE,
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL
#         )
#         print(f"RTSPStreamer: publishing to {self.rtsp_url}")

#     def write(self, frame):
#         if self.proc is None or self.proc.stdin is None:
#             return

#         if frame is None:
#             return

#         if frame.shape[1] != self.width or frame.shape[0] != self.height:
#             frame = cv2.resize(frame, (self.width, self.height))

#         try:
#             with self.lock:
#                 self.proc.stdin.write(frame.tobytes())
#         except Exception:
#             pass

#     def stop(self):
#         try:
#             if self.proc and self.proc.stdin:
#                 self.proc.stdin.close()
#         except Exception:
#             pass

#         try:
#             if self.proc:
#                 self.proc.terminate()
#         except Exception:
#             pass

#         self.proc = None
#         print("RTSPStreamer: stopped")

class RTSPStreamer:
    def __init__(self, rtsp_url, width, height, fps=20):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None
        self.lock = threading.Lock()

    def start(self):
        cmd = [
            FFMPEG_BIN,
            "-re",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # print(f"RTSPStreamer: publishing to {self.rtsp_url}")
            print(f"RTSPStreamer: publishing to {self.rtsp_url}")
        except FileNotFoundError:
            print("RTSPStreamer error: ffmpeg executable not found")
            print("Check FFMPEG_BIN path")
            self.proc = None
            return
    def write(self, frame):
        if self.proc is None or self.proc.stdin is None:
            return

        if frame is None:
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
# THREADED PROCESSOR
# Processes only newest frame and drops stale work
# =========================================================
class InspectionProcessor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.input_lock = threading.Lock()
        self.latest_input = None
        self.latest_result = None
        self.result_lock = threading.Lock()

        self.smoother = VerdictSmoother(
            window=SMOOTH_WINDOW,
            pass_thresh=PASS_MAJORITY,
            reject_thresh=REJECT_MAJORITY,
        )

        self.processed_count = 0
        self.last_proc_time = 0.0

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

            crop_frame, _, _ = center_crop(frame, CROP_W, CROP_H)
            enhanced_frame, gray, otsu, adap, prep_info = preprocess_variants(crop_frame)
            display_frame = enhanced_frame.copy()

            display_verdict = "NO"
            display_pts = None
            display_error_type = "no_qr"
            display_metrics = {}
            qr_data = ""

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
                ok, pts = detect_qr_shape(gray)
                if ok:
                    display_verdict = "REJECT"
                    display_pts = pts
                    display_error_type = "unreadable_qr"
                    display_metrics = compute_quality_metrics(enhanced_frame, pts)
                else:
                    display_verdict = "NO"
                    display_error_type = "no_qr"

            final_verdict = self.smoother.update(display_verdict)

            if display_verdict != "NO":
                draw_result(display_frame, display_verdict, display_pts, display_error_type, display_metrics)

            # payload = make_payload(
            #     verdict=final_verdict,
            #     error_type=display_error_type,
            #     qr_data=qr_data,
            #     pts=display_pts,
            #     metrics=display_metrics,
            #     prep_info=prep_info
            # )
            # send_payload_to_clients(payload)
                        # AFTER (correct)
            payload = make_payload(
                verdict=final_verdict,
                error_type=display_error_type,
                qr_data=qr_data,
                pts=display_pts,
                metrics=display_metrics,
                prep_info=prep_info
            )

            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            payload["frame"] = base64.b64encode(buffer).decode('utf-8')

            send_payload_to_clients(payload)

            proc_ms = (time.perf_counter() - t0) * 1000.0
            self.last_proc_time = proc_ms
            self.processed_count += 1

            result = {
                "frame_id": frame_id,
                "display_frame": display_frame,
                "final_verdict": final_verdict,
                "display_error_type": display_error_type,
                "qr_data": qr_data,
                "prep_info": prep_info,
                "payload": payload,
                "proc_ms": proc_ms,
                "processed_count": self.processed_count,
            }

            with self.result_lock:
                self.latest_result = result


# =========================================================
# MAIN WINDOW
# =========================================================
class InspectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("inspection_gui.ui", self)

        self.streamer = None
        self.stream_width = CROP_W
        self.stream_height = CROP_H

        # self.rtsp_url = RTSP_URL
        self.reader = None
        self.processor = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)

        self.startButton.clicked.connect(self.start_camera)
        self.stopButton.clicked.connect(self.stop_camera)

        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setScaledContents(False)
        self.messageLabel.setAlignment(Qt.AlignCenter)
        self.messageLabel.setWordWrap(True)

        self.last_gui_time = time.perf_counter()
        self.last_seen_reader_frame_id = -1
        self.last_seen_result_frame_id = -1
        self.ui_frame_count = 0

        self.messageLabel.setText("Waiting for camera...")

#    def start_camera(self):
        # self.stop_camera()

        # self.reader = RTSPReader(self.rtsp_url)
        # self.processor = InspectionProcessor()

        # self.reader.start()
        # self.processor.start()

        # time.sleep(0.3)

        # if not self.reader.connected:
        #     self.messageLabel.setText("RTSP camera failed to open")
        #     self.reader = None
        #     self.processor = None
        #     return

        # self.last_gui_time = time.perf_counter()
        # self.last_seen_reader_frame_id = -1
        # self.last_seen_result_frame_id = -1
        # self.ui_frame_count = 0

        # self.timer.start(30)
        # self.messageLabel.setText("Camera started (threaded mode)")

    # def start_camera(self):
    #     self.stop_camera()

    #     self.reader = RTSPReader(self.rtsp_url)
    #     self.processor = InspectionProcessor()

    #     self.reader.start()

    #     # wait up to 5 seconds for RTSP open result
    #     self.reader.ready_event.wait(timeout=5.0)

    #     if not self.reader.connected:
    #         self.messageLabel.setText("RTSP camera failed to open")
    #         self.reader = None
    #         self.processor = None
    #         return

    #     self.processor.start()

    #     self.last_gui_time = time.perf_counter()
    #     self.last_seen_reader_frame_id = -1
    #     self.last_seen_result_frame_id = -1
    #     self.ui_frame_count = 0

    #     self.timer.start(30)
    #     self.messageLabel.setText("Camera started (threaded mode)")


    def start_camera(self):
        self.stop_camera()

        self.reader = USBCameraReader(
            camera_index=USB_CAMERA_INDEX,
            width=USB_WIDTH,
            height=USB_HEIGHT,
            fps=USB_FPS
        )
        self.processor = InspectionProcessor()

        self.reader.start()

        # wait up to 5 seconds for USB camera open result
        self.reader.ready_event.wait(timeout=5.0)

        if not self.reader.connected:
            self.messageLabel.setText("USB camera failed to open")
            self.reader = None
            self.processor = None
            return

        self.processor.start()

        # start RTSP publisher for processed frames
        self.streamer = RTSPStreamer(
            rtsp_url=RTSP_PUBLISH_URL,
            width=self.stream_width,
            height=self.stream_height,
            fps=20
        )
        self.streamer.start()

        self.last_gui_time = time.perf_counter()
        self.last_seen_reader_frame_id = -1
        self.last_seen_result_frame_id = -1
        self.ui_frame_count = 0

        self.timer.start(30)
        self.messageLabel.setText("USB camera started and RTSP streaming")

    # def stop_camera(self):
    #     self.timer.stop()

    #     if self.reader is not None:
    #         self.reader.stop()
    #         self.reader = None

    #     if self.processor is not None:
    #         self.processor.stop()
    #         self.processor = None

    #     self.imageLabel.clear()
    #     self.imageLabel.setText("Real Image")
    #     self.messageLabel.setText("Camera stopped")

    def stop_camera(self):
        self.timer.stop()

        if self.reader is not None:
            self.reader.stop()
            self.reader = None

        if self.processor is not None:
            self.processor.stop()
            self.processor = None

        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None

        self.imageLabel.clear()
        self.imageLabel.setText("Real Image")
        self.messageLabel.setText("Camera stopped")

    def set_message(self, text, verdict):
        if verdict == "PASS":
            bg = "rgb(0, 190, 90)"
        elif verdict == "REJECT":
            bg = "rgb(220, 70, 70)"
        else:
            bg = "rgb(25, 156, 214)"

        self.messageLabel.setStyleSheet(
            f"color: black; font: 16pt 'Arial'; font-weight: bold; background-color: {bg};"
        )
        self.messageLabel.setText(text)

    def update_gui(self):
        if self.reader is None or self.processor is None:
            return

        # always fetch newest camera frame
        frame, frame_id = self.reader.get_latest_frame()

        # submit only if frame is newer
        if frame is not None and frame_id != self.last_seen_reader_frame_id:
            self.last_seen_reader_frame_id = frame_id
            self.processor.submit_frame(frame, frame_id)

        # display newest processed result
        result = self.processor.get_latest_result()
        if result is None:
            return

        result_frame_id = result["frame_id"]
        if result_frame_id == self.last_seen_result_frame_id:
            return

        self.last_seen_result_frame_id = result_frame_id
        self.ui_frame_count += 1

        display_frame = result["display_frame"]
        if self.streamer is not None:
            self.streamer.write(display_frame)
        final_verdict = result["final_verdict"]
        display_error_type = result["display_error_type"]
        qr_data = result["qr_data"]
        prep_info = result["prep_info"]
        proc_ms = result["proc_ms"]

        if final_verdict == "PASS":
            text = f"Inspection Result: PASS\nData: {qr_data if qr_data else '-'}"
        elif final_verdict == "REJECT":
            text = f"Inspection Result: REJECT\nReason: {display_error_type}"
        else:
            text = "Inspection Result: NO QR DETECTED"

        self.set_message(text, final_verdict)

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        label_w = max(1, self.imageLabel.width())
        label_h = max(1, self.imageLabel.height())
        resized = cv2.resize(rgb, (label_w, label_h))

        h, w, ch = resized.shape
        bytes_per_line = ch * w
        qt_image = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.imageLabel.setPixmap(QPixmap.fromImage(qt_image))

        now = time.perf_counter()
        gui_fps = 1.0 / max(now - self.last_gui_time, 1e-9)
        self.last_gui_time = now

        retinex_text = "ON" if prep_info["used_retinex"] else "OFF"
        self.statusBar().showMessage(
            f"GUI FPS: {gui_fps:.1f} | Reader Frame ID: {self.last_seen_reader_frame_id} | "
            f"Shown Result ID: {result_frame_id} | Verdict: {final_verdict} | "
            f"Retinex: {retinex_text} | Proc: {proc_ms:.1f} ms | WS Port: {WS_PORT}"
        )

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()

    app_qt = QApplication(sys.argv)
    window = InspectionWindow()
    window.show()
    sys.exit(app_qt.exec_())