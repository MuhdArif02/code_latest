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
from fastapi.responses import HTMLResponse
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
# DASHBOARD HTML
# =========================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QR Inspection Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0c10;
    --panel: #10141c;
    --border: #1e2535;
    --accent-pass: #00dc50;
    --accent-reject: #ff2d2d;
    --accent-no: #00a8ff;
    --text: #c8d4e8;
    --muted: #4a5568;
    --mono: 'Share Tech Mono', monospace;
    --sans: 'Barlow', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }

  /* scanline overlay */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.07) 2px,
      rgba(0,0,0,0.07) 4px
    );
    pointer-events: none;
    z-index: 999;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .logo-icon {
    width: 32px;
    height: 32px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: 1fr 1fr 1fr;
    gap: 3px;
  }

  .logo-icon span {
    background: var(--accent-no);
    border-radius: 1px;
    animation: pulse-icon 2s ease-in-out infinite;
  }

  .logo-icon span:nth-child(2), .logo-icon span:nth-child(4), .logo-icon span:nth-child(6) {
    background: transparent;
    border: 1px solid var(--accent-no);
  }

  @keyframes pulse-icon {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .logo h1 {
    font-family: var(--mono);
    font-size: 16px;
    letter-spacing: 2px;
    color: var(--accent-no);
    text-transform: uppercase;
  }

  .logo span {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1px;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 20px;
  }

  .ws-status {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }

  .ws-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--muted);
    transition: background 0.3s;
  }

  .ws-dot.connected { background: var(--accent-pass); box-shadow: 0 0 8px var(--accent-pass); }
  .ws-dot.disconnected { background: var(--accent-reject); }

  .controls {
    display: flex;
    gap: 8px;
  }

  .btn {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 1px;
    padding: 6px 16px;
    border: 1px solid;
    border-radius: 2px;
    cursor: pointer;
    background: transparent;
    text-transform: uppercase;
    transition: all 0.15s;
  }

  .btn-start {
    border-color: var(--accent-pass);
    color: var(--accent-pass);
  }

  .btn-start:hover {
    background: var(--accent-pass);
    color: #000;
  }

  .btn-stop {
    border-color: var(--accent-reject);
    color: var(--accent-reject);
  }

  .btn-stop:hover {
    background: var(--accent-reject);
    color: #fff;
  }

  /* ---- MAIN LAYOUT ---- */
  .main {
    display: grid;
    grid-template-columns: 1fr 340px;
    grid-template-rows: auto 1fr;
    gap: 1px;
    flex: 1;
    background: var(--border);
  }

  /* ---- VERDICT BANNER ---- */
  .verdict-banner {
    grid-column: 1 / -1;
    background: var(--panel);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 72px;
    position: relative;
    overflow: hidden;
    transition: background 0.4s;
  }

  .verdict-banner::after {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--current-color, var(--accent-no));
    transition: background 0.3s;
  }

  .verdict-text {
    font-family: var(--mono);
    font-size: 26px;
    font-weight: bold;
    letter-spacing: 4px;
    transition: color 0.3s;
  }

  .verdict-sub {
    font-size: 13px;
    color: var(--muted);
    margin-top: 3px;
    font-family: var(--mono);
    letter-spacing: 1px;
  }

  .verdict-qr {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text);
    text-align: right;
    max-width: 420px;
    word-break: break-all;
  }

  .verdict-qr .label {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  /* ---- CAMERA FEED ---- */
  .camera-panel {
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    min-height: 400px;
  }

  .camera-panel img {
    max-width: 100%;
    max-height: 100%;
    display: block;
    object-fit: contain;
  }

  .camera-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 12px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 13px;
    pointer-events: none;
  }

  .camera-overlay.hidden { display: none; }

  .spinner {
    width: 36px;
    height: 36px;
    border: 2px solid var(--border);
    border-top-color: var(--accent-no);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* corner brackets on camera */
  .corner {
    position: absolute;
    width: 24px;
    height: 24px;
    border-color: var(--muted);
    border-style: solid;
    opacity: 0.4;
  }
  .corner.tl { top: 12px; left: 12px; border-width: 2px 0 0 2px; }
  .corner.tr { top: 12px; right: 12px; border-width: 2px 2px 0 0; }
  .corner.bl { bottom: 12px; left: 12px; border-width: 0 0 2px 2px; }
  .corner.br { bottom: 12px; right: 12px; border-width: 0 2px 2px 0; }

  /* ---- METRICS PANEL ---- */
  .metrics-panel {
    background: var(--panel);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }

  .panel-section {
    padding: 18px 20px;
    border-bottom: 1px solid var(--border);
  }

  .panel-section h3 {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .metric-label {
    font-size: 12px;
    color: var(--muted);
    font-family: var(--mono);
    letter-spacing: 0.5px;
  }

  .metric-value {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text);
  }

  .metric-bar-wrap {
    margin-bottom: 12px;
  }

  .metric-bar-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
    font-family: var(--mono);
    font-size: 11px;
  }

  .metric-bar-header .name { color: var(--muted); }
  .metric-bar-header .val { color: var(--text); }

  .bar-track {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease, background 0.3s;
  }

  .tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid;
  }

  .tag-pass   { color: var(--accent-pass);   border-color: var(--accent-pass);   background: rgba(0,220,80,0.08); }
  .tag-reject { color: var(--accent-reject); border-color: var(--accent-reject); background: rgba(255,45,45,0.08); }
  .tag-no     { color: var(--accent-no);     border-color: var(--accent-no);     background: rgba(0,168,255,0.08); }
  .tag-on     { color: var(--accent-pass);   border-color: var(--accent-pass);   background: rgba(0,220,80,0.08); }
  .tag-off    { color: var(--muted);         border-color: var(--muted);         background: transparent; }

  /* ---- STATUS BAR ---- */
  .statusbar {
    grid-column: 1 / -1;
    background: var(--panel);
    border-top: 1px solid var(--border);
    padding: 6px 24px;
    display: flex;
    gap: 28px;
    align-items: center;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.5px;
  }

  .statusbar .item { display: flex; gap: 6px; }
  .statusbar .item .k { color: var(--border); }
  .statusbar .item .v { color: var(--text); }

  /* verdict color states */
  .state-pass  { --current-color: var(--accent-pass);   color: var(--accent-pass);   }
  .state-reject{ --current-color: var(--accent-reject); color: var(--accent-reject); }
  .state-no    { --current-color: var(--accent-no);     color: var(--accent-no);     }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">
      <span></span><span></span><span></span>
      <span></span><span></span><span></span>
      <span></span><span></span><span></span>
    </div>
    <div>
      <h1>QR Inspect</h1>
      <span>Visual Inspection System</span>
    </div>
  </div>
  <div class="header-right">
    <div class="ws-status">
      <div class="ws-dot disconnected" id="wsDot"></div>
      <span id="wsLabel">DISCONNECTED</span>
    </div>
    <div class="controls">
      <button class="btn btn-start" onclick="startCamera()">▶ Start</button>
      <button class="btn btn-stop" onclick="stopCamera()">■ Stop</button>
    </div>
  </div>
</header>

<div class="main">

  <!-- VERDICT BANNER -->
  <div class="verdict-banner state-no" id="verdictBanner">
    <div>
      <div class="verdict-text" id="verdictText">WAITING</div>
      <div class="verdict-sub" id="verdictSub">Connect camera to begin inspection</div>
    </div>
    <div class="verdict-qr" id="verdictQr" style="display:none">
      <div class="label">Decoded Data</div>
      <div id="qrDataText"></div>
    </div>
  </div>

  <!-- CAMERA FEED -->
  <div class="camera-panel" id="cameraPanel">
    <div class="corner tl"></div>
    <div class="corner tr"></div>
    <div class="corner bl"></div>
    <div class="corner br"></div>
    <div class="camera-overlay" id="cameraOverlay">
      <div class="spinner"></div>
      <span>Awaiting feed...</span>
    </div>
    <img id="cameraFeed" src="" alt="" style="display:none; width:100%; height:100%; object-fit:contain;">
  </div>

  <!-- METRICS PANEL -->
  <div class="metrics-panel">

    <div class="panel-section">
      <h3>Inspection Result</h3>
      <div class="metric-row">
        <span class="metric-label">Status</span>
        <span class="tag tag-no" id="statusTag">NO QR</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Error Type</span>
        <span class="metric-value" id="errorType">—</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Confidence</span>
        <span class="metric-value" id="confidence">—</span>
      </div>
    </div>

    <div class="panel-section">
      <h3>Image Quality</h3>

      <div class="metric-bar-wrap">
        <div class="metric-bar-header">
          <span class="name">Blur (Laplacian)</span>
          <span class="val" id="blurVal">—</span>
        </div>
        <div class="bar-track"><div class="bar-fill" id="blurBar" style="width:0%;background:var(--accent-no)"></div></div>
      </div>

      <div class="metric-bar-wrap">
        <div class="metric-bar-header">
          <span class="name">Dynamic Range</span>
          <span class="val" id="dynVal">—</span>
        </div>
        <div class="bar-track"><div class="bar-fill" id="dynBar" style="width:0%;background:var(--accent-no)"></div></div>
      </div>

      <div class="metric-bar-wrap">
        <div class="metric-bar-header">
          <span class="name">Bimodal Ratio</span>
          <span class="val" id="bimodalVal">—</span>
        </div>
        <div class="bar-track"><div class="bar-fill" id="bimodalBar" style="width:0%;background:var(--accent-no)"></div></div>
      </div>

      <div class="metric-bar-wrap">
        <div class="metric-bar-header">
          <span class="name">Solidity</span>
          <span class="val" id="solidityVal">—</span>
        </div>
        <div class="bar-track"><div class="bar-fill" id="solidityBar" style="width:0%;background:var(--accent-no)"></div></div>
      </div>

      <div class="metric-bar-wrap">
        <div class="metric-bar-header">
          <span class="name">Mean Saturation</span>
          <span class="val" id="satVal">—</span>
        </div>
        <div class="bar-track"><div class="bar-fill" id="satBar" style="width:0%;background:var(--accent-no)"></div></div>
      </div>
    </div>

    <div class="panel-section">
      <h3>Geometry</h3>
      <div class="metric-row">
        <span class="metric-label">Angle (raw)</span>
        <span class="metric-value" id="angleVal">—</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Angle (norm)</span>
        <span class="metric-value" id="normAngleVal">—</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Size (W × H)</span>
        <span class="metric-value" id="sizeVal">—</span>
      </div>
    </div>

    <div class="panel-section">
      <h3>Preprocessing</h3>
      <div class="metric-row">
        <span class="metric-label">Retinex</span>
        <span class="tag tag-off" id="retinexTag">OFF</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Brightness</span>
        <span class="metric-value" id="brightnessVal">—</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Contrast (σ)</span>
        <span class="metric-value" id="contrastVal">—</span>
      </div>
    </div>

    <div class="panel-section">
      <h3>Session Stats</h3>
      <div class="metric-row">
        <span class="metric-label">Total Frames</span>
        <span class="metric-value" id="frameCount">0</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">PASS</span>
        <span class="metric-value" id="passCount" style="color:var(--accent-pass)">0</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">REJECT</span>
        <span class="metric-value" id="rejectCount" style="color:var(--accent-reject)">0</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">NO QR</span>
        <span class="metric-value" id="noCount" style="color:var(--accent-no)">0</span>
      </div>
    </div>

  </div><!-- /metrics-panel -->

</div><!-- /main -->

<div class="statusbar" id="statusbar">
  <div class="item"><span class="k">WS</span><span class="v" id="sbWs">—</span></div>
  <div class="item"><span class="k">FPS</span><span class="v" id="sbFps">—</span></div>
  <div class="item"><span class="k">LATENCY</span><span class="v" id="sbLatency">—</span></div>
  <div class="item"><span class="k">PORT</span><span class="v">""" + str(WS_PORT) + """</span></div>
</div>

<script>
  let ws = null;
  let stats = { pass: 0, reject: 0, no: 0, total: 0 };
  let lastFrameTime = null;
  let fpsBuffer = [];

  const WS_URL = `ws://${location.hostname}:""" + str(WS_PORT) + """/ws`;

  function connectWS() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setWsDot(true);
      document.getElementById('sbWs').textContent = WS_URL;
    };

    ws.onclose = () => {
      setWsDot(false);
      setTimeout(connectWS, 2000);
    };

    ws.onerror = () => setWsDot(false);

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        updateDashboard(data);
      } catch(e) {}
    };
  }

  function setWsDot(connected) {
    const dot = document.getElementById('wsDot');
    const lbl = document.getElementById('wsLabel');
    dot.className = 'ws-dot ' + (connected ? 'connected' : 'disconnected');
    lbl.textContent = connected ? 'CONNECTED' : 'DISCONNECTED';
  }

  function updateDashboard(d) {
    const verdict = d.status || 'NO';
    const metrics = d.metrics || {};
    const now = performance.now();

    // FPS
    if (lastFrameTime !== null) {
      const fps = 1000 / (now - lastFrameTime);
      fpsBuffer.push(fps);
      if (fpsBuffer.length > 10) fpsBuffer.shift();
      const avgFps = fpsBuffer.reduce((a,b)=>a+b,0) / fpsBuffer.length;
      document.getElementById('sbFps').textContent = avgFps.toFixed(1);
    }
    lastFrameTime = now;

    // Latency
    const latencyMs = (Date.now() / 1000 - d.timestamp) * 1000;
    document.getElementById('sbLatency').textContent = latencyMs.toFixed(0) + 'ms';

    // Stats
    stats.total++;
    if (verdict === 'PASS') stats.pass++;
    else if (verdict === 'REJECT') stats.reject++;
    else stats.no++;
    document.getElementById('frameCount').textContent = stats.total;
    document.getElementById('passCount').textContent = stats.pass;
    document.getElementById('rejectCount').textContent = stats.reject;
    document.getElementById('noCount').textContent = stats.no;

    // Camera feed
    if (d.frame) {
      const img = document.getElementById('cameraFeed');
      img.src = 'data:image/jpeg;base64,' + d.frame;
      img.style.display = 'block';
      document.getElementById('cameraOverlay').classList.add('hidden');
    }

    // Verdict banner
    const banner = document.getElementById('verdictBanner');
    const vText = document.getElementById('verdictText');
    const vSub = document.getElementById('verdictSub');
    const vQr = document.getElementById('verdictQr');
    const qrDataText = document.getElementById('qrDataText');

    banner.className = 'verdict-banner state-' + verdict.toLowerCase();
    if (verdict === 'NO') banner.className += ' state-no';

    if (verdict === 'PASS') {
      vText.textContent = '✓ PASS';
      vSub.textContent = 'QR Code accepted';
      if (d.qr_data) {
        vQr.style.display = 'block';
        qrDataText.textContent = d.qr_data;
      }
    } else if (verdict === 'REJECT') {
      vText.textContent = '✗ REJECT';
      vSub.textContent = 'Reason: ' + (d.error_type || '—');
      vQr.style.display = 'none';
    } else {
      vText.textContent = '— NO QR';
      vSub.textContent = 'No QR code detected in frame';
      vQr.style.display = 'none';
    }

    // Status tag
    const tag = document.getElementById('statusTag');
    tag.textContent = verdict === 'NO' ? 'NO QR' : verdict;
    tag.className = 'tag tag-' + verdict.toLowerCase();

    document.getElementById('errorType').textContent = d.error_type || '—';
    document.getElementById('confidence').textContent =
      d.confidence != null ? (d.confidence * 100).toFixed(1) + '%' : '—';

    // Quality bars
    function setBar(barId, valId, value, max, warn, good) {
      const pct = Math.min(100, (value / max) * 100);
      const bar = document.getElementById(barId);
      bar.style.width = pct + '%';
      const color = value >= good ? 'var(--accent-pass)' :
                    value >= warn ? '#ffaa00' : 'var(--accent-reject)';
      bar.style.background = color;
      document.getElementById(valId).textContent = value != null ? value.toFixed ? value.toFixed(1) : value : '—';
    }

    setBar('blurBar',    'blurVal',    metrics.blur    || 0, 200,  55,  80);
    setBar('dynBar',     'dynVal',     metrics.dynamic || 0, 255,  35,  80);
    setBar('bimodalBar', 'bimodalVal', metrics.bimodal || 0, 1,    0.45, 0.6);
    setBar('solidityBar','solidityVal',metrics.solidity|| 0, 1,    0.75, 0.9);
    setBar('satBar',     'satVal',     metrics.sat     || 0, 100,  70,  70);

    // Geometry
    document.getElementById('angleVal').textContent =
      metrics.angle != null ? metrics.angle.toFixed(1) + '°' : '—';
    document.getElementById('normAngleVal').textContent =
      metrics.norm_angle != null ? metrics.norm_angle.toFixed(1) + '°' : '—';
    document.getElementById('sizeVal').textContent =
      (metrics.w && metrics.h) ? metrics.w + ' × ' + metrics.h + ' px' : '—';

    // Preprocessing
    const rx = d.retinex_used;
    const rtag = document.getElementById('retinexTag');
    rtag.textContent = rx ? 'ON' : 'OFF';
    rtag.className = 'tag ' + (rx ? 'tag-on' : 'tag-off');

    document.getElementById('brightnessVal').textContent =
      d.brightness != null ? d.brightness.toFixed(1) : '—';
    document.getElementById('contrastVal').textContent =
      d.contrast != null ? d.contrast.toFixed(1) : '—';
  }

  function startCamera() {
    fetch('/camera/start', { method: 'POST' })
      .then(r => r.json())
      .then(d => console.log('Start:', d.status));
  }

  function stopCamera() {
    fetch('/camera/stop', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        console.log('Stop:', d.status);
        document.getElementById('cameraFeed').style.display = 'none';
        document.getElementById('cameraOverlay').classList.remove('hidden');
        stats = { pass: 0, reject: 0, no: 0, total: 0 };
        fpsBuffer = [];
        lastFrameTime = null;
      });
  }

  connectWS();
</script>
</body>
</html>
"""


# =========================================================
# WEBSOCKET SERVER (FastAPI)
# =========================================================
app = FastAPI(title="QR Inspection Dashboard")
connected_clients = set()

# Global handles for camera/processor/streamer
_reader = None
_processor = None
_streamer = None


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


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

        time.sleep(0.015)  # ~66 fps poll


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
def compute_quality_metrics(frame, pts):
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    enhanced, used_retinex, _, _ = auto_retinex_if_needed(frame)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray_roi = get_safe_roi(gray, x, y, w, h)
    bgr_roi  = get_safe_roi(enhanced, x, y, w, h)

    if gray_roi.size == 0 or bgr_roi.size == 0:
        return {"blur": 0.0, "dynamic": 0, "bimodal": 0.0, "solidity": 0.0,
                "angle": 0.0, "norm_angle": 0.0, "sat": 0.0, "w": w, "h": h, "retinex": used_retinex}

    blur    = measure_blur(gray_roi)
    dynamic = int(gray_roi.max()) - int(gray_roi.min())
    total   = gray_roi.size
    extreme = int(np.sum(gray_roi < 64)) + int(np.sum(gray_roi > 192))
    bimodal = extreme / max(total, 1)
    hsv     = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mean_sat= float(hsv[:, :, 1].mean())
    solidity= polygon_solidity(pts)
    angle   = get_angle_from_pts(pts)
    norm_angle = normalize_qr_angle(angle)

    return {"blur": blur, "dynamic": dynamic, "bimodal": bimodal, "solidity": solidity,
            "angle": angle, "norm_angle": norm_angle, "sat": mean_sat,
            "w": w, "h": h, "retinex": used_retinex}


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
        self.processed_count = 0
        self.last_proc_time  = 0.0

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

            display_verdict    = "NO"
            display_pts        = None
            display_error_type = "no_qr"
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
                ok, pts = detect_qr_shape(gray)
                if ok:
                    display_verdict    = "REJECT"
                    display_pts        = pts
                    display_error_type = "unreadable_qr"
                    display_metrics    = compute_quality_metrics(enhanced_frame, pts)
                else:
                    display_verdict    = "NO"
                    display_error_type = "no_qr"

            final_verdict = self.smoother.update(display_verdict)

            if display_verdict != "NO":
                draw_result(display_frame, display_verdict, display_pts,
                            display_error_type, display_metrics)

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
