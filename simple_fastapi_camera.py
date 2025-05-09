import cv2
import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from collections import deque

# Depth Anything用
from depth_processor import DepthProcessor, create_depth_visualization, create_depth_grid_visualization

app = FastAPI()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

depth_processor = DepthProcessor()

# 処理時間計測用
camera_times = deque(maxlen=1000)
inference_times = deque(maxlen=1000)
visualization_times = deque(maxlen=1000)
encoding_times = deque(maxlen=1000)

def log_processing_times():
    """5秒ごとに平均、最大、最小の処理時間をログ出力"""
    while True:
        time.sleep(5)
        if camera_times:
            print(f"[Camera] Avg: {np.mean(camera_times):.4f}s, Max: {np.max(camera_times):.4f}s, Min: {np.min(camera_times):.4f}s")
        if inference_times:
            print(f"[Inference] Avg: {np.mean(inference_times):.4f}s, Max: {np.max(inference_times):.4f}s, Min: {np.min(inference_times):.4f}s")
        if visualization_times:
            print(f"[Visualization] Avg: {np.mean(visualization_times):.4f}s, Max: {np.max(visualization_times):.4f}s, Min: {np.min(visualization_times):.4f}s")
        if encoding_times:
            print(f"[Encoding] Avg: {np.mean(encoding_times):.4f}s, Max: {np.max(encoding_times):.4f}s, Min: {np.min(encoding_times):.4f}s")

import threading
threading.Thread(target=log_processing_times, daemon=True).start()

def get_depth_stream():
    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read()
        camera_times.append(time.perf_counter() - start_time)
        if not ret:
            time.sleep(0.001)
            continue

        start_time = time.perf_counter()
        small = cv2.resize(frame, (224, 224))
        depth_map, _ = depth_processor.predict(small)
        inference_times.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        vis = create_depth_visualization(depth_map, small.shape)
        vis = cv2.resize(vis, (320, 240))
        visualization_times.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoding_times.append(time.perf_counter() - start_time)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.001)

def get_camera_stream():
    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read()
        camera_times.append(time.perf_counter() - start_time)
        if not ret:
            time.sleep(0.001)
            continue

        start_time = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoding_times.append(time.perf_counter() - start_time)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.001)

def get_depth_grid_stream():
    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read()
        camera_times.append(time.perf_counter() - start_time)
        if not ret:
            time.sleep(0.001)
            continue

        start_time = time.perf_counter()
        small = cv2.resize(frame, (224, 224))
        depth_map, _ = depth_processor.predict(small)
        inference_times.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        grid_img = create_depth_grid_visualization(depth_map, grid_size=(12, 16), cell_size=18)
        if grid_img is None or len(grid_img.shape) < 2:
            grid_img = 128 * np.ones((240, 320, 3), dtype=np.uint8)
        elif len(grid_img.shape) == 2 or (len(grid_img.shape) == 3 and grid_img.shape[2] == 1):
            grid_img = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
        grid_img = cv2.resize(grid_img, (320, 240))
        visualization_times.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', grid_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoding_times.append(time.perf_counter() - start_time)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.001)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head><title>Fast Camera Streaming</title></head>
    <body>
        <h2>Camera Stream</h2>
        <img src="/video" alt="Camera Stream" />
        <h2>Depth Anything (Depth Map)</h2>
        <img src="/depth_video" alt="Depth Map" />
        <h2>Depth Grid</h2>
        <img src="/depth_grid" alt="Depth Grid" />
    </body>
    </html>
    """

@app.get("/video")
async def video():
    return StreamingResponse(get_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_video")
async def depth_video():
    return StreamingResponse(get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_grid")
async def depth_grid():
    return StreamingResponse(get_depth_grid_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)