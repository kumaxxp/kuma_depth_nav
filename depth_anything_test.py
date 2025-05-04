#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Anything テストプログラム

1. GUIの無いLinux環境でカメラ画像を撮影してブラウザに表示
2. ユーザーがブラウザ上のボタンを押すとその時の画像を撮影
3. 撮影画像をDepth Anythingで変換して深度画像を取得
4. 撮影画像と深度画像をブラウザに表示
"""

import cv2
import numpy as np
import threading
import time
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn

# Depth Anythingモデル用にaxengineをインポート（存在する場合）
try:
    import axengine as axe
    AXENGINE_AVAILABLE = True
    print("[INFO] axengine is available. Depth Anything can be used.")
except ImportError:
    AXENGINE_AVAILABLE = False
    print("[WARNING] axengine is not installed. Depth Anything cannot be used.")

# FastAPIアプリケーションの初期化
app = FastAPI()

# グローバル変数
latest_frame = None
latest_depth = None
frame_lock = threading.Lock()
depth_lock = threading.Lock()
capture_event = threading.Event()
stop_event = threading.Event()

# モデルパスのデフォルト値
DEFAULT_MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'
depth_model = None
depth_model_input_name = None

# カメラ初期化
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("[ERROR] Failed to open the camera.")

# モデル初期化
def initialize_depth_model(model_path):
    global depth_model, depth_model_input_name

    if not AXENGINE_AVAILABLE:
        print("[ERROR] axengine is not available.")
        return False

    try:
        print(f"[INFO] Loading model from {model_path}")
        options = {
            "axe.input_layout": "NHWC",
            "axe.output_layout": "NHWC",
            "axe.use_dsp": "true"
        }
        depth_model = axe.InferenceSession(model_path, options)
        depth_model_input_name = depth_model.get_inputs()[0].name
        print("[INFO] Depth model loaded successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize depth model: {e}")
        return False

# Depth Anythingモデルの初期化
if AXENGINE_AVAILABLE:
    initialize_depth_model(DEFAULT_MODEL_PATH)

# カメラフレーム取得スレッド
def camera_thread():
    global latest_frame

    while not stop_event.is_set():
        ret, frame = camera.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(0.03)  # 約30FPS

# 深度推論
def run_depth_inference(frame):
    if depth_model is None:
        return None

    try:
        resized = cv2.resize(frame, (384, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)
        outputs = depth_model.run(None, {depth_model_input_name: input_tensor})
        return outputs[0] if outputs else None
    except Exception as e:
        print(f"[ERROR] Depth inference failed: {e}")
        return None

# ストリーミング用のジェネレーター関数
def video_stream():
    global latest_frame, latest_depth

    while not stop_event.is_set():
        frame = None
        depth_vis = None

        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()

        if frame is not None:
            with depth_lock:
                if latest_depth is not None:
                    depth_vis = latest_depth.copy()

            # 深度マップがない場合はプレースホルダーを作成
            if depth_vis is None:
                depth_vis = np.zeros_like(frame)
                cv2.putText(depth_vis, "No Depth", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 両方の画像を横に並べる
            combined = np.hstack((frame, depth_vis))
            _, buffer = cv2.imencode('.jpg', combined)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# 画像キャプチャエンドポイント
@app.get("/capture")
async def capture():
    global latest_frame, latest_depth

    with frame_lock:
        if latest_frame is None:
            return JSONResponse(content={"error": "No frame available."})
        frame = latest_frame.copy()

    depth_map = run_depth_inference(frame)

    if depth_map is not None:
        depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255.0 / depth_map.max()), cv2.COLORMAP_JET)
        with depth_lock:
            latest_depth = depth_vis.copy()

    return JSONResponse(content={"message": "Image captured and depth map updated."})

# メインページ
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Depth Anything Test</title>
        </head>
        <body>
            <h1>Depth Anything Test</h1>
            <img src="/video" style="width: 100%; max-width: 800px;" />
            <button onclick="captureImage()">Capture Image</button>
            <script>
                function captureImage() {
                    fetch('/capture').then(response => response.json()).then(data => {
                        alert(data.message || data.error);
                    });
                }
            </script>
        </body>
    </html>
    """

# ビデオストリームエンドポイント
@app.get("/video")
async def video():
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

# メイン関数
def main():
    camera_thread_instance = threading.Thread(target=camera_thread, daemon=True)
    camera_thread_instance.start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
    finally:
        stop_event.set()
        camera_thread_instance.join()
        camera.release()

if __name__ == "__main__":
    main()