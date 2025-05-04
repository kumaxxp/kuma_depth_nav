from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import subprocess

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>USBカメラ ストリーム</title>
        </head>
        <body>
            <h1>USBカメラの映像を表示中</h1>
            <img src="/video" width="640" height="480" />
        </body>
    </html>
    """

def initialize_camera(index=0, width=640, height=480):
    # v4l2-ctl を使ってサポートされる基本的な設定のみを適用
    try:
        subprocess.run(["v4l2-ctl", f"--set-fmt-video=width={width},height={height},pixelformat=MJPG"], check=False)
        subprocess.run(["v4l2-ctl", "--set-parm=15"], check=False)
    except Exception as e:
        print(f"[WARN] Failed to apply v4l2-ctl settings: {e}")

    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def get_video_stream():
    camera = initialize_camera()
    times = []
    last_report = time.time()

    try:
        while True:
            start_time = time.perf_counter()

            if not camera.isOpened():
                print("[WARN] Camera not open. Retrying...")
                camera.release()
                time.sleep(0.5)
                camera = initialize_camera()
                continue

            success, frame = camera.read()
            if not success or frame is None:
                print("[WARN] Failed to read frame. Skipping...")
                continue

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("[WARN] JPEG encode failed.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

            if time.time() - last_report >= 5.0:
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    min_time = min(times)
                    fps = len(times) / 5.0
                    print(f"[PERF] Avg: {avg_time:.4f}s, Max: {max_time:.4f}s, Min: {min_time:.4f}s, FPS: {fps:.1f}")
                    times.clear()
                last_report = time.time()

            #time.sleep(0.005)

    finally:
        camera.release()

@app.get("/video")
async def video_endpoint():
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
